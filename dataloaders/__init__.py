import os
from dataloaders.datasets import voc, visdrone, visdrone_chip
from torch.utils.data import DataLoader


def make_data_loader(opt, hyp, train=True):

    if opt.dataset in ['visdrone', 'Visdrone', 'VisDrone']:
        if train:
            root_name = "VisDrone2019-DET-train"
        else:
            root_name = "VisDrone2019-DET-val"

        # Dataset
        dataset = visdrone.VisdroneDataset(
            os.path.join(opt.root_path, root_name),
            opt.img_size, hyp=hyp, train=train)

        # Dataloader
        dataloader = DataLoader(dataset,
                                batch_size=opt.batch_size,
                                num_workers=min([os.cpu_count(), opt.batch_size, 16]),
                                shuffle=True,  # Shuffle=True unless rectangular training is used
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)


        return dataset, dataloader