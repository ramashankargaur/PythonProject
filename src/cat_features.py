from sklearn import preprocessing

class catFeatures:
    def __init__(self, df, cat_features, encoding_type, handle_nan=False):
        """
        df: pandas dataframe
        cat_features: Name of header in the data set i.e. "bin_0, bin_1, bin_2, bin_3, bin_4, nom_0, nom_1, nom_2, nom_3, etc
        encoding_type: It's a predefined theortical values i.e. label, binary, one hot encoding, etc
        handle_nan: 'nan' values within each features of the data set which is boolean i.e. True/False
        """
        self.df = df
        self.df_new = self.df.copy(deep=True)
        self.cat_features = cat_features
        self.encoding_type = encoding_type
        self.handle_nan = handle_nan
        self.lbl_encoders = dict()
        self.bin_encoders = dict()

        if self.handle_nan:
            for c in self.cat_features:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-9999")

    def _label_encoding(self):
        for c in self.cat_features:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.df_new.loc[:, c] = lbl.transform(self.df[c].values)
            self.lbl_encoders[c] = lbl
        return self.df_new

    def _label_binarization(self):
        for c in self.cat_features:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val_bin = lbl.transform(self.df[c].values)
            self.df_new = self.df_new.drop(c, axis=1)
            for j in range(val_bin.shape[1]):
                bin_col_name = c + f"_new_{j}"
                self.df_new[bin_col_name] = val_bin[:, j]
            self.bin_encoders[c] = lbl
        return self.df_new

    def transform_new(self): #fit_transform
        if self.encoding_type == "label":
            return self._label_encoding()
        elif self.encoding_type == "binary":
            return self._label_binarization()
        else:
            raise Exception("Encoding type not understood..!!")

    def my_transform(self, dataframe): #transform
        if self.handle_nan:
            for c in self.cat_features:
                dataframe.loc[:, c] = dataframe.loc[:, c].astype(str).fillna("-9999")

        if self.encoding_type == "label":
            for c, lbl in self.lbl_encoders.items():
                dataframe.loc[:, c] = lbl.my_transform(dataframe[c].values)
            return dataframe

        elif self.encoding_type == "binary":
            for c in self.bin_encoders.items():
                val_bin = lbl.my_transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)
                for j in range(val_bin.shape[1]):
                    bin_col_name = c + f"_new_{j}"
                    dataframe[bin_col_name] = val_bin[:, j]
            return dataframe

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("../input/train_categoricalFeature.csv").head(500)
    #df_test = pd.read_csv("../input/test_categoricalFeature.csv").head(500)
    cat_cols = [c for c in df.columns if c not in ["id", "target"]]
    print(cat_cols)
    cat_features = catFeatures(df, 
                                cat_features=cat_cols,
                                encoding_type="binary",
                                handle_nan=True)
    df_new = cat_features.transform_new()
    #train_transformed = cat_features.transform_new()
    #test_transformed = cat_features.my_transform(df_test)
    print(df_new.head())
    #print(test_transformed)
