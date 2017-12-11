# update: 2017-05-22

class BaseModel(object):
    '''Base Model
    '''
    def __init__(self, df, y_var, x_vars, has_const=True):
        self.df = df
        self.y_var = y_var
        self.x_vars = x_vars.copy()
        self.has_const = has_const
        self._fitted = False

    def _clean_data(self):
        reg_vars = [self.y_var] + self.x_vars if self.y_var is not None else self.x_vars
        # reg_df only contains valid variables
        self.reg_df = self.df[reg_vars].dropna()
        if self.has_const:
            if '_const' not in self.reg_df: self.reg_df['_const'] = 1
            if '_const' not in self.x_vars: self.x_vars.append('_const')
        
    def fit(self):
        raise NotImplementedError

    def predict(self, df):
        raise NotImplementedError
        