if __name__ == '__main__': #évite des problem de broken pipe, merci à tanujjain sur https://github.com/idealo/imagededup/issues/67

    import pandas as pd
    from fastai.vision.all import *
    from fastai.data.core import *
    import os
    #load data
    data_folder = "data"

    train_df = pd.read_csv(os.path.join(data_folder, "train.csv"))
    #Visualise data
    train_df

    train_df['ImageID'] = train_df['ImageID'].astype(str)+".jpg"
    train_df
    #Training phase, create model
    dls = ImageDataLoaders.from_df(train_df, path=os.path.join(data_folder, "train"), bs=8, y_block=RegressionBlock)
    dls.show_batch()

    learn = cnn_learner(dls, alexnet, metrics=mse)
    #Train the model
    learn.fine_tune(1)
    #Test phase, Loading test set
    test_imgs_name = get_image_files(os.path.join(data_folder, "test"))
    test_dls = dls.test_dl(test_imgs_name)

    test_img_ids = [re.sub(r"\D", "", str(img_name)) for img_name in test_imgs_name]

    test_dls.show_batch()
    #Predict test set
    _,_,results = learn.get_preds(dl = test_dls, with_decoded = True)

    results = [i[0] for i in results.numpy()]
    #Save the prediction to csv
    submission = pd.DataFrame({"ImageID":test_img_ids, "label":results})
    submission

    submission.to_csv("submission.csv", index=False)
