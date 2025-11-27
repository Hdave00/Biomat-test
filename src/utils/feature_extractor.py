"""
Yet another script that I made but didnt end up using, due to limited computing power as im using my CPU to train the models. The whole point of this feature
extractor was to make use of it as a helper in a ML script for a unified dataset comprised of multiple data files but the complex architecture requirements 
needed, in order to not overfit the model, on training the relationships between the features of each CSV, proved to be too much. 

In the future, with better hardware this is entirely plausible to develop & and the end result of a model that has learned the relationships between key
datapoints from each csv mentioned below, would be an amazing generalization for Regression tasks and even complex relationship plotting tasks, to predict
what material, with what type of micro-structures is suitably non-toxic for which bio-environment.
"""

# utils/feature_extractor.py
import pandas as pd

class FeatureExtractor:
    def __init__(self, mechanical_df, chemical_df, biological_df, master_index):
        self.mechanical = mechanical_df
        self.chemical = chemical_df
        self.biological = biological_df
        self.master_index = master_index

    def get_material_features(self, material_name):
        name_norm = material_name.strip().upper()
        if name_norm not in self.master_index['material_name'].values:
            return None

        # Mechanical
        mech = self.mechanical[self.mechanical['Material'].str.upper() == name_norm]
        mech_features = mech.drop(columns=['Material'], errors='ignore').mean().fillna(-1).to_dict()

        # Chemical
        chem = self.chemical[self.chemical['Polymer'].str.upper() == name_norm]
        chem_features = chem.drop(columns=['Polymer'], errors='ignore').mean().fillna(-1).to_dict()

        # Biological
        bio = self.biological[self.biological['Material'].str.upper() == name_norm]
        bio_features = bio.drop(columns=['Material'], errors='ignore').mean().fillna(-1).to_dict()

        return {
            'mechanical': mech_features,
            'chemical': chem_features,
            'biological': bio_features
        }

    def get_training_matrix(self):
        """Return X, y ready for neural network training"""
        X_mech, X_chem, X_bio, y = [], [], [], []

        for _, row in self.master_index.iterrows():
            feats = self.get_material_features(row['material_name'])
            if feats is None:
                continue

            # Skip if insufficient data
            if not feats['mechanical'] or not feats['chemical'] or not feats['biological']:
                continue

            X_mech.append(list(feats['mechanical'].values()))
            X_chem.append(list(feats['chemical'].values()))
            X_bio.append(list(feats['biological'].values()))

            # Example: biocompatibility score from cytotoxicity
            bio_score = feats['biological'].get('cytotoxicity_score', 0.5)
            y.append(bio_score)

        return X_mech, X_chem, X_bio, y