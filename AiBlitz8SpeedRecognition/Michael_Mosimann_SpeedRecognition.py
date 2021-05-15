if __name__ == '__main__': #évite des problem de broken pipe, merci à tanujjain sur https://github.com/idealo/imagededup/issues/67

    import pandas as pd
    from fastai.vision.all import *
    from fastai.data.core import *
    import os
    #from PIL import Image
    #from tqdm.notebook import tqdm
    # load data
    data_folder = "data"

    train_df = pd.read_csv(os.path.join(data_folder, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_folder, "val.csv"))
    # Visualise data
    # train_df
    train_df['ImageID'] = train_df['ImageID'].astype(str)+".jpg"
    val_df['ImageID'] = val_df['ImageID'].astype(str)+".jpg"
    # train_df
    print(train_df)
    # converting the images to gray to see if results will improve.
    data_directiory = "data"
    train_path = os.path.join(data_directiory, "train")
    train_submission_path = os.path.join(data_directiory, "train_gray")
    test_path = os.path.join(data_directiory, "test")
    test_submission_path = os.path.join(data_directiory, "test_gray")
    # Grayscaling
    #for img_name in tqdm(os.listdir(train_path)):
        # Opening the image and transforming in grayscale
    #    img = Image.open(os.path.join(train_path, f"{img_name}")).convert("L")
        # Saving the output image
    #    img.save(os.path.join(train_submission_path, f"{img_name}"))
    #for img_name in tqdm(os.listdir(test_path)):
        # Opening the image and transforming in grayscale
    #    img = Image.open(os.path.join(test_path, f"{img_name}")).convert("L")
        # Saving the output image
    #    img.save(os.path.join(test_submission_path, f"{img_name}"))
    # Training phase, create model
    #dls = ImageDataLoaders.from_df(train_df, path=os.path.join(data_folder, "train"), bs=8, y_block=RegressionBlock)
    dls = ImageDataLoaders.from_df(train_df, path=os.path.join(data_folder, "train_gray"), bs=8, y_block=RegressionBlock)
    dls.show_batch()
    #dls_val = ImageDataLoaders.from_df(val_df, path=os.path.join(data_folder, "val"), bs=8, y_block=RegressionBlock)
    #dls_val.show_batch()
    learn = cnn_learner(dls, alexnet, metrics=mse)  # model creation/loading.
    learn.summary()
    # learn = cnn_learner(dls, resnet18, metrics=mse)
    # Train the model
    learn.fine_tune(1)
    # Test phase, Loading test set
    test_imgs_name = get_image_files(os.path.join(data_folder, "test_gray"))
    test_dls = dls.test_dl(test_imgs_name)

    test_img_ids = [re.sub(r"\D", "", str(img_name)) for img_name in test_imgs_name]

    test_dls.show_batch()
    # Predict test set
    _, _, results = learn.get_preds(dl = test_dls, with_decoded = True)

    results = [i[0] for i in results.numpy()]
    # Save the prediction to csv
    submission = pd.DataFrame({"ImageID": test_img_ids, "label": results})
    submission

    submission.to_csv("submission.csv", index=False)
