import argparse
import logging
from ssd.config.defaults import cfg
from ssd.engine.inference import do_evaluation
from ssd.modeling.detector import SSDDetector
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.logger import setup_logger
from ssd import torch_utils
from ssd.data.build import make_data_loader
from torchvision.utils import save_image


def create_feature_maps(cfg, ckpt):
    logger = logging.getLogger("SSD.inference")

    model = SSDDetector(cfg)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR, logger=logger)
    model = torch_utils.to_cuda(model)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    """
    dataset_list = cfg.DATASETS.TEST
    dataset = build_dataset(cfg.DATASET_DIR, dataset_list, is_train=False)
    """
    data_loaders = make_data_loader(cfg, is_train=False)

    # TODO: clean up these loops
    for dl in data_loaders:
        data_loader = dl
        break

    dataset = data_loader.dataset

    
    image = None
    for im in dataset:
        print("image length:", len(im))
        image = im
        break

    features = []
    for image in images:
        features.append(model.backbone(image))
        break
    

    return images, features

def visualize_feature_maps(images, features):

    for layer in features:
        for channel in layer[0]:
            torchvision.utils.save_image(
                channel[0][-1],
                f"visualization/layer{layer}_channel{channel}.png")


def main():
    parser = argparse.ArgumentParser(description='SSD Visulization of Feature Maps')
    parser.add_argument(
        "config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for visualization, default is the latest checkpoint.",
        default=None,
        type=str,
    )
    parser.add_argument("--output_dir", default="feature_map_visualization", type=str, help="The directory to store visualizations")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    """
    logger = setup_logger("SSD", cfg.OUTPUT_DIR)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    """
    images, features = create_feature_maps(cfg, ckpt=args.ckpt)
    visualize_feature_maps(images,features)

if __name__ == '__main__':
    main()