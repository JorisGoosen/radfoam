import argparse
import os
import warnings

import pycolmap
from PIL import Image
import tqdm
import shutil


def main(args):
    """Minimal script to run COLMAP on a directory of images."""

    data_dir = args.data_dir
    reconstruction_dir = os.path.join(data_dir, "sparse")

    alreadyReconstructed = False
    biggestOne=0

    reconstruction = None
    if os.path.exists(reconstruction_dir):
        alreadyReconstructed = True
        #raise ValueError("Reconstruction directory already exists")
        reconstruction = pycolmap.Reconstruction(reconstruction_dir)

    images_dir = os.path.join(data_dir, "images")
    if not os.path.exists(images_dir):
        raise ValueError("data_dir must contain an 'images' directory")

    if not alreadyReconstructed:
        database_path = os.path.join(data_dir, "database.db")
        if os.path.exists(database_path):
            raise ValueError("Database file already exists")

        database = pycolmap.Database(database_path)

        pycolmap.extract_features(
            database_path,
            images_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            camera_model=args.camera_model,
        )

        print(f"Imported {database.num_images} images to {database_path}")

        pycolmap.match_exhaustive(database_path, sift_options=pycolmap.SiftMatchingOptions(guided_matching=True))

        print(f"Feature matching completed")

        os.makedirs(reconstruction_dir)

        reconstructions = pycolmap.incremental_mapping(
            database_path,
            image_path=images_dir,
            output_path=reconstruction_dir,
        )

        reconstruction = reconstructions[0]

        if len(reconstructions) > 1:
            warnings.warn("Multiple reconstructions found, taking biggest one")
            maxL =  reconstruction.num_images()
            for r in range(len(reconstructions)):

                if reconstructions[r].num_images() > maxL:
                    maxL = reconstructions[r].num_images()
                    reconstruction = reconstructions[r]
                    biggestOne = r
    
    print("Biggest one was " + str(biggestOne))
    rechterPad = os.path.join(data_dir, "rechter")
    os.makedirs(rechterPad, exist_ok=True)    
    inames=[]
    allinames=[]
    for image in tqdm.tqdm(list(reconstruction.images.values())):
        inames.append(image.name)
    
    for r in range(len(reconstructions)):
        for image in tqdm.tqdm(list(reconstructions[r].images.values())):
            if not image.name in allinames:
                allinames.append(image.name)
        rechteImgs = os.path.join(data_dir, "rechter" + str(r))
        pycolmap.undistort_images(output_path=rechteImgs, input_path=os.path.join(data_dir, "sparse", str(r)), image_path=os.path.join(data_dir, "images"), image_names=inames)
        dir_path = os.path.join(rechteImgs, "images")
        for file_path in os.listdir(dir_path):
            src = os.path.join(dir_path, file_path)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(rechterPad, file_path))
        

    os.makedirs(os.path.join(data_dir, "images_2"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "images_4"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "images_8"), exist_ok=True)

    print("Downsampling images")

    baseW = -1
    baseH = -1

    for imagename in allinames:
        image_1_path = os.path.join(rechterPad, imagename)
        image_2_path = os.path.join(data_dir, "images_2", imagename)
        image_4_path = os.path.join(data_dir, "images_4", imagename)
        image_8_path = os.path.join(data_dir, "images_8", imagename)

        pil_image = Image.open(image_1_path)
        if baseW == -1:
            baseH = pil_image.height
            baseW = pil_image.width

        pil_image_2 = pil_image.resize(
            (baseW // 2, baseH // 2),
            resample=Image.LANCZOS,
        )
        pil_image_4 = pil_image.resize(
            (baseW // 4, baseH // 4),
            resample=Image.LANCZOS,
        )
        pil_image_8 = pil_image.resize(
            (baseW // 8, baseH // 8),
            resample=Image.LANCZOS,
        )

        pil_image_2.save(image_2_path)
        pil_image_4.save(image_4_path)
        pil_image_8.save(image_8_path)

        pil_image.close()
        pil_image_2.close()
        pil_image_4.close()
        pil_image_8.close()

    print("Exporting point cloud")

    reconstruction.export_PLY(os.path.join(data_dir, "point_cloud.ply"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--camera_model", type=str, default="OPENCV")
    args = parser.parse_args()
    main(args)
