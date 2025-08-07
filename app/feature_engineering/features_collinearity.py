import pandas as pd
import numpy as np

class Multicollinearity:
    def __init__(self, df, target_columns = 'close', threshold = 0.8):
        self.df = df.copy()
        self.threshold = threshold
        self.target_columns = target_columns
        if target_columns is not None and target_columns in self.df.columns:
            self.features_df = self.df.drop(columns = [target_columns])
        else:
            self.features_df = self.df

        self.features_df = self.features_df.select_dtypes(include=[np.number])

    
    def correlation_matrix(self):
        return self.features_df.corr()
    
    def get_correlated_features(self):
        corr_matrix = self.correlation_matrix().abs()
        # Mask self correlations and duplicates
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_pairs = []
        for col in upper_tri.columns:
            for row in upper_tri.index:
                corr_val = upper_tri.at[row, col]
                if pd.notna(corr_val) and corr_val>self.threshold:
                    corr_pairs.append((row,col, corr_val))
        return sorted(corr_pairs, key=lambda x: -x[2])
    
    # def report(self):
    #     pairs = self.get_correlated_features()
    #     if not pairs:
    #         print(f"No pairs of features found with correlation above {self.threshold}")
    #         return
    #     print(f"Features with correlation above {self.threshold}:")
    #     for f1, f2, corr_val in pairs:
    #         print(f"  {f1} <--> {f2} : correlation = {corr_val:.3f}")

    def features_to_drop(self):
        to_drop = set()
        target_corr = self.df.corr()[self.target_columns] if self.target_columns else None
        pairs = self.get_correlated_features()

        for f1, f2, _ in pairs:
            if f1 in to_drop or f2 in to_drop:
                continue

            if self.target_columns:
                corr1 = abs(target_corr.get(f1, 0))
                corr2 = abs(target_corr.get(f2, 0))
                if corr1 > corr2:
                    to_drop.add(f2)
                    continue
                elif corr2 > corr1:
                    to_drop.add(f1)
                    continue
        return to_drop
    
    def report(self):
        print(f"Checking multicollinearity with threshold = {self.threshold}")
        pairs = self.get_correlated_features()
        if not pairs:
            print("No highly correlated feature pairs found.")
            return

        print(f"{len(pairs)} highly correlated feature pairs found:")
        for f1, f2, corr in pairs:
            print(f"  {f1} <--> {f2} | correlation = {corr:.3f}")
        
        drops = self.features_to_drop()
        print("\n🧹 Suggested features to drop:")
        for f in sorted(drops):
            print(f"  - {f}")
