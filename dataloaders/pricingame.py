
import numpy as np
import pandas as pd
from dataloaders import BaseDataset, Cat, Num
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DATA_URL = None


VARIABLES = {
    "total_claim_amount": Num, 
    "pol_bonus": Num, 
    "pol_duration": Num, 
    "pol_sit_duration": Num, 
    "pol_insee_code": Num, 
    "drv_age1": Num,          
    "drv_age2": Num, 
    "drv_age_lic1": Num, 
    "drv_age_lic2": Num, 
    "vh_age": Num, 
    "vh_cyl": Num,
    "vh_din": Num,
    "vh_make": Num,
    "vh_model": Num, 
    "vh_sale_begin": Num, 
    "vh_sale_end": Num, 
    "vh_speed": Num, 
    "vh_value": Num,
    "vh_weight": Num, 
    "pol_coverageMaxi": Num, 
    "pol_coverageMedian1": Num, 
    "pol_coverageMedian2": Num, 
    "pol_coverageMini": Num, 
    "pol_pay_freqMonthly": Num  ,
    "pol_pay_freqQuarterly": Num, 
    "pol_pay_freqYearly": Num, 
    "pol_paydYes": Num, 
    "pol_usageProfessional": Num, 
    "pol_usageRetired": Num,  
    "pol_usageWorkPrivate": Num ,
    "vh_fuelGasoline": Num,  
    "vh_fuelHybrid": Num, 
    "vh_typeTourism": Num,
    "drv_sex1": Num
}


class PricinGameDataset(BaseDataset):
    def __init__(self, download=True, data_dir=None, random_seed=0):

        super().__init__(
            download=download,
            data_dir=data_dir,
            random_seed=random_seed
        )

        self._outcome_type = 'continuous'
        self._sensitive_attr_name = 'drv_sex1'
        self.process()

    def process(self):
        data_path = self.data_dir.joinpath('pricingame.data')

        self._check_exists_and_download(data_path, DATA_URL, self.download)
        
        data = pd.read_csv(data_path, sep=',')
        gender = data[self._sensitive_attr_name]
        target = data['total_claim_amount']
        
        dat = data.drop([self._sensitive_attr_name, 'total_claim_amount', "pol_insee_code", "vh_make", "vh_model"], axis=1)
        
        train_X, test_X, train_A, test_A, train_y, test_y = train_test_split(
            dat, gender, target, test_size=0.3, random_state=self.random_seed)
        
        self._train = pd.DataFrame(train_X,
                                   index=train_X.index, columns=train_X.columns)
        self._test = pd.DataFrame(test_X,
                                  index=test_X.index, columns=test_X.columns)
        
        self._train['drv_sex1'] = train_A
        self._train['target'] = train_y
        self._test['drv_sex1'] = test_A
        self._test['target'] = test_y
        
        
if __name__ == '__main__':
    data = PricinGameDataset(data_dir='../datafiles')
    print(data.train.shape)
    print(data.test.shape)