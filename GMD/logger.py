from pathlib import Path

import torch


class Logger(object):
    def __init__(self, log_root, tag, topic_dir, ckpt_dir):
        log_root = Path(log_root)
        log_dir = log_root / tag
        topic_dir, ckpt_dir = log_dir / topic_dir, log_dir / ckpt_dir
        log_root.mkdir(exist_ok=True)
        try:
            log_dir.mkdir(exist_ok=tag == 'default')
        except FileExistsError:
            raise FileExistsError(f'{log_dir} already exists. Please specify a different tag for the new run.')
        topic_dir.mkdir(exist_ok=True)
        ckpt_dir.mkdir(exist_ok=True)

        self._log_dir = log_dir
        self._topic_dir = topic_dir
        self._ckpt_dir = ckpt_dir
        self._log_file = log_dir / 'run.log'

    def log(self, msg):
        print(msg)
        with self._log_file.open('a') as f:
            f.write(msg + '\n')

    def save_text(self, topics, filename):
        with open(self._topic_dir / filename, 'w') as f:
            f.write(topics)

    def save_model(self, model, ckpt_name):
        torch.save(model, self._ckpt_dir / ckpt_name)
