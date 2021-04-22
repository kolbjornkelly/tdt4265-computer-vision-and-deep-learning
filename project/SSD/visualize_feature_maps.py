import argparse
import logging
from ssd.config.defaults import cfg
from ssd.engine.inference import do_evaluation
from ssd.modeling.detector import SSDDetector
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.logger import setup_logger
from ssd import torch_utils
from ssd.data.datasets import build_dataset
from torchvision.utils import save_image


def create_feature_maps(cfg, ckpt, num_visualizations):
    logger = logging.getLogger("SSD.inference")

    model = SSDDetector(cfg)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR, logger=logger)
    model = torch_utils.to_cuda(model)
    checkpointer.load(ckpt, use_latest=ckpt is None)

    dataset_list = cfg.DATASET.TEST
    dataset = build_dataset(cfg.DATASET_DIR, dataset_list, is_train=False):

    images = dataset[0:num_visualizations]
    features = []
    for image in images:
        features.append(model.backbone(image))

    return images, features

def visualize_feature_maps(images, features):

    for layer in features:
        for channel in ftr[0]:
            torchvision.utils.save_image(
                channel[0][-1],
                f"visualization/layer{layer}_channel{channel}.png")




def main(num_visualizations)
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

    logger = setup_logger("SSD", cfg.OUTPUT_DIR)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    images, features = create_feature_maps(cfg, ckpt=args.ckpt, num_visualizations)
    visualize_feature_maps(images,features)

if __name__ == '__main__':
    main(num_visualizations)