"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
import collections
from openai import OpenAI
OPENAI_API_KEY = ""
import pdb
from tqdm import trange
import re
import numpy as np


@registry.register_task("captioning")
class CaptionTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True, dataset_name=None):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric
        self.dataset_name = dataset_name

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
            dataset_name=list(cfg.datasets_cfg.keys())[0],
        )

    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        img_ids = samples["image_id"]
        text_outputs = samples['text_output']
        for i, (caption, img_id) in enumerate(zip(captions, img_ids)):
            if 'coco' in self.dataset_name:
                results.append({"caption": caption, "image_id": int(img_id)})
            else:
                results.append({"caption": caption, "image_id": img_id, 'label': text_outputs[i]})

        return results

    def after_evaluation(self, val_result, split_name, epoch, dataset, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        if self.report_metric:
            if 'coco' in self.dataset_name:
                metrics = self._report_metrics_coco(
                    eval_result_file=eval_result_file, split_name=split_name
                )
            else:
                metrics = self._report_metrics_video_caption(
                    eval_result_file=eval_result_file, split_name=split_name, dataset=dataset, dataset_name=self.dataset_name
                )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics_coco(self, eval_result_file, split_name):

        # TODO better way to define this
        coco_gt_root = os.path.join(registry.get_path("cache_root"), "coco_gt")
        coco_val = coco_caption_eval(coco_gt_root, eval_result_file, split_name)

        agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
        log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res

    def extract_scores(self, judge_generation):
        max_score = 10
        ratings = [
            int(d)
            for d in re.findall(
                "[0-9]+",
                judge_generation.strip()
                .splitlines()[-1]
                .replace(f"/{max_score}", "")
                .replace(f"out of {max_score}", ""),
            )
        ]
        ratings = list((filter(lambda x: 1 <= x <= max_score, ratings)))
        if len(ratings) != 1:
            ratings = [1]
            # raise ValueError(f"Could not parse single integer from judge output: {judge_generation}")
        
        # judge_generation['score'] = ratings[0]
        return {"score": (ratings[0]-1)/9}


    def evaluate_gpt4(self, generation, label):
        prompt = f"""
        [Groundtruth]\n
        {label}\n\n
        [Assistant Answer]\n
        {generation}\n\n
        [System]\n\n
        We would like to request your feedback whether AI assistant answer semantically matches groundtruth. Don't get too strict in rating. You should also give credit to partial semantic overlap.\n\n
        Please first output a paragraph assessing the helpfulness, relevance and accuracy of the answer. In the subsequent line, please output your score as a single integer on a scale of 1 to 10.
        """
        response = self.client.chat.completions.create(
            model='gpt-4-1106-preview',
            messages=[
                {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
                {"role": "user", "content": prompt}
            ]
        )
        eval_output = response.choices[0].message.content
        score = self.extract_scores(eval_output)['score']
        return score

    @main_process
    def _report_metrics_video_caption(self, eval_result_file, split_name, dataset, dataset_name):
        filenames = {
            "train": "cap_train.json",
            "val": "cap_val.json",
            "test": "cap_test.json",
        }
        if 'msvd' in dataset_name:
            annotation_file = os.path.join(registry.get_path("cache_root"), "msvd/annotation/coco_gt", filenames[split_name])
        elif 'msrvtt' in dataset_name:
            annotation_file = os.path.join(registry.get_path("cache_root"), "msrvtt/annotation/coco_gt", filenames[split_name])
        elif 'youcook2' in dataset_name:
            annotation_file = os.path.join(registry.get_path("cache_root"), "youcook2/annotation/coco_gt", filenames[split_name])
        elif 'vidal' in dataset_name:
            annotation_file = os.path.join(registry.get_path("cache_root"), "vidal/annotation/coco_gt", filenames[split_name])

        try:
            self.client = OpenAI(
                api_key=OPENAI_API_KEY
            )
            with open(eval_result_file, 'r') as f: eval_result_list = json.load(f)
            with open(annotation_file, 'r') as f: annotation_list = json.load(f)['annotations']

            score_list = []
            for i in trange(len(eval_result_list)):
                generation = eval_result_list[i]['caption']
                label = eval_result_list[i]['label']
                eval_output = self.evaluate_gpt4(generation, label)
                score_list.append(eval_output)
            
            score = np.array(score_list)
            top_1 = np.mean(score)
            res = {}
            res["agg_metrics"] = top_1
        except:
            coco = COCO(annotation_file)
            coco_result = coco.loadRes(eval_result_file)

            # create coco_eval object by taking coco and coco_result
            coco_eval = COCOEvalCap(coco, coco_result)

            # evaluate results
            # SPICE will take a few minutes the first time, but speeds up due to caching
            coco_eval.evaluate()

            # print output evaluation scores
            for metric, score in coco_eval.eval.items():
                print(f"{metric}: {score:.3f}")

            agg_metrics = coco_eval.eval["CIDEr"] + coco_eval.eval["Bleu_4"] + coco_eval.eval["METEOR"] + coco_eval.eval["ROUGE_L"]
            log_stats = {split_name: {k: v for k, v in coco_eval.eval.items()}}

            res = {k: v for k, v in coco_eval.eval.items()}
            res["agg_metrics"] = agg_metrics


        return res



# TODO better structure for this.
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_url


def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {
        "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
        "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
    }
    filenames = {
        "val": "coco_karpathy_val_gt.json",
        "test": "coco_karpathy_test_gt.json",
    }

    download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames[split])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval
