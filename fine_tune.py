import json
from pathlib import Path

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard

import nets


class TensorBoardWithLR(TensorBoard):
    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


def infer_args_from_weights_path(wpath):
    [basenet, _, batch_size, _] = wpath.split('/')[-4:]
    return basenet, int(batch_size)


class FTCNN(object):

    def __init__(self, basenet="xception"):
        self.basenet = basenet
        self.img_target_size = nets.IMAGE_TARGET_SIZE[basenet]
        self.preprocess_func = nets.PREPROCESS_FUNC[basenet]
        self.model = None
        self.batch_size = None

        self.n_classes = None

    def _get_generator(self, img_dir, batch_size, test_mode=False):
        datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess_func,
        )
        if test_mode:
            class_mode = None
        else:
            class_mode = 'categorical'

        gen = datagen.flow_from_directory(
            str(img_dir),
            target_size=self.img_target_size,
            batch_size=batch_size,
            class_mode=class_mode,
        )
        return gen

    def _get_ckpt_cb(self):
        ckpt_dir = self.run_dir / 'ckpt' / self.project_name / \
                self.basenet / self.optim / str(self.batch_size)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = ckpt_dir

        ckpt_file = ckpt_dir / \
            'weights.ep{epoch:02d}' \
            '-{val_acc:.2f}' \
            '-{val_top_k_categorical_accuracy:.2f}.hdf5'
        return ModelCheckpoint(filepath=str(ckpt_file))

    def _get_tensorboard_cb(self):
        tb_path = self.run_dir / 'tensorboard' / self.project_name / \
                self.basenet / self.optim / str(self.batch_size)
        return TensorBoardWithLR(log_dir=str(tb_path))

    def train(
            self,
            dataset_root,
            run_dir,
            project_name,
            optim="adam",
            batch_size=32,
            epochs=50,
            ):

        self.dataset_root = Path(dataset_root)
        self.train_root = self.dataset_root / 'train'
        self.val_root = self.dataset_root / 'val'

        self.run_dir = Path(run_dir)
        self.project_name = project_name

        self.optim = optim

        self.batch_size = batch_size
        train_gen = self._get_generator(self.train_root, batch_size)
        val_gen = self._get_generator(self.val_root, batch_size)
        assert train_gen.class_indices == val_gen.class_indices
        cls2id = dict([(v, k) for (k, v) in train_gen.class_indices.items()])
        print(cls2id)
        print(train_gen.class_indices)

        self.n_classes = len(train_gen.class_indices)
        print('class #: %d' % self.n_classes)

        self.mckpt = self._get_ckpt_cb()
        self.tb = self._get_tensorboard_cb()
        json.dump(cls2id, (self.ckpt_dir / 'id2class.json').open('w'))

        self.model = nets.get_model(
            self.n_classes,
            basenet=self.basenet,
            optim=self.optim,
        )

        self.model.fit_generator(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=[self.mckpt, self.tb]
        )

    def predict(self, weights_path, test_root, batch_size, top=3):

        cls_json = Path(weights_path).parent / 'id2class.json'
        id2cls = json.load(cls_json.open())
        self.n_classes = len(id2cls)

        self.model = nets.get_model(
            self.n_classes,
            basenet=self.basenet,
        )
        self.model.load_weights(weights_path)

        test_gen = self._get_generator(test_root, batch_size, test_mode=True)
        res = self.model.predict_generator(test_gen)
        for e, fn in zip(res, test_gen.filenames):
            idices = e.argsort()[::-1][:top]
            lbls = [(id2cls[str(i)], e[i]) for i in idices]
            lbls.sort(key=lambda x: x[1], reverse=True)
            print(fn)
            for l in lbls:
                print('%s[%.2f]' % l)
        return res


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-basenet", default="xception")
    parser.add_argument("-dataset", help="dataset root")
    parser.add_argument(
            "-run_dir", help="directory for checkpoints and TensorBoard")
    parser.add_argument("-name", help="project name")
    parser.add_argument("-optim", default="adam", help="optimizer")
    parser.add_argument("-batch_size", type=int, default=32, help="batch size")
    parser.add_argument("-epochs", type=int, default=50, help="num of epochs")

    parser.add_argument("-test_root", default=None, help="test images root")
    parser.add_argument("-weights", help="weights path")

    args = parser.parse_args()

    if not args.test_root:
        ftcnn = FTCNN(args.basenet)
        ftcnn.train(
                args.dataset,
                run_dir=args.run_dir,
                project_name=args.name,
                optim=args.optim,
                batch_size=args.batch_size,
                epochs=args.epochs,
                )
    else:
        basenet, batch_size = infer_args_from_weights_path(args.weights)
        ftcnn = FTCNN(basenet)
        ftcnn.predict(args.weights, args.test_root, batch_size)
