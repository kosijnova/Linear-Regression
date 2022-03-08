class WrongColumnsNames(Exception):
    pass

class WrongLengthError(Exception):
    pass

class LinearRegression(object):
    def __init__(self, df, y_names, x_names):
        self.df = df
        self.y_names = y_names
        self.x_names = x_names
    
    #Prepare data to LSM
    @staticmethod
    def data_prep(df, x_names, y_names):
        try:
            colnames = y_names + x_names
            if all(x in df.columns.values.tolist() for x in colnames) == False:
                raise WrongColumnsNames
        except WrongColumnsNames:
            print("Check columns names!")
            return 
        else:
            df = df[colnames]
            alpha = pd.DataFrame(np.ones(len(df)), columns = ['alpha'])
            df = pd.concat([alpha, df], axis = 1)
        return df
    
    @staticmethod
    def lsm(df, x_names, y_names):
        y = df[y_names].to_numpy()
        x = df[['alpha'] + x_names].to_numpy()
        x_t = np.transpose(x)
        #first part of formula
        f_part = np.matmul(x_t, x)
        f_part = np.linalg.inv(f_part)
        
        #second part of formula
        s_part = np.matmul(x_t, y)
        
        #out
        out = np.transpose(np.matmul(f_part, s_part))
        
        return out
    
    @staticmethod
    def y_tar(x, out):
        y_tar = np.sum(np.multiply(x, out), axis = 1)
        y_tar = np.array([y_tar]).T
        return y_tar
    
    @staticmethod
    def remainder(y, y_tar):
        remainder = y - y_tar
        return remainder.to_numpy()
    
    @staticmethod
    def table(y, y_tar, remainder):
        y_hat = pd.DataFrame(y, columns = ['y_hat'])
        y_tar = pd.DataFrame(y_tar, columns = ['y_tar'])
        remainder = pd.DataFrame(remainder, columns = ['remainder'])
        table = pd.concat([y_hat, y_tar, remainder], axis = 1)
        
        return table
    
    @staticmethod
    def coef_of_determination(y, y_tar):
        y = y.to_numpy()
        avg = np.mean(y)
        down = np.sum(np.power(np.subtract(y, avg), 2))
        up = np.sum(np.power(np.subtract(y_tar, avg), 2))
        coef = up/down
        
        return coef
    
    def fit(self):
        data = self.data_prep(self.df, self.x_names, self.y_names)
        
        self.lsm = self.lsm(data, self.x_names, self.y_names)
        
        y_hat = data[self.y_names].to_numpy()
        
        x = data[['alpha'] + self.x_names].to_numpy()
        y_tar = self.y_tar(x, self.lsm)
        
        remainder = self.remainder(data[self.y_names], y_tar)
        
        coef = self.coef_of_determination(data[self.y_names], y_tar)
        
        FIT = namedtuple("FIT", "COEF y_hat y_tar remainder R2")
        fit = FIT(self.lsm, y_hat, y_tar, remainder, coef)
        
        return fit
    
    def predict(self, x):
        try:
            if len(self.lsm[0])-1 != x.shape[1]:
                raise WrongLengthError
        except WrongLengthError:
            print("Wrong Vector Length!")
            return
        else:
            ones = np.ones(len(x))
            x = np.insert(x, 0, np.ones(len(x)).tolist(), axis = 1)
            y_tar = np.sum(np.multiply(x, self.lsm), axis = 1)
            y_tar = np.array([y_tar]).T
        
        return y_tar
        
if __name__ == "__main__":
    from collections import namedtuple
    import pandas as pd
    import numpy as np
    
    df = pd.read_excel('xxx')
    kmnk = LinearRegression(df, ['Y'], ['X1'])
    fit = kmnk.fit()
    
    print(fit, end = '\n')
    print(kmnk.table(fit.y_hat, fit.y_tar, fit.remainder), end = '\n')
    print(kmnk.predict(np.array([[12],[13],[14]])))
