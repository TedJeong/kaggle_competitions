# -*- encoding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import time
import operator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss, f1_score, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns



trn = pd.read_csv('../input/train_append_lb_lag.csv').fillna(0)
target = pd.DataFrame(pickle.load(open('../input/target.pkl','rb')), columns=['target'])
tst = pd.read_csv('../input/test_append_lb_lag.csv').fillna(0)
print(trn.shape, target.shape, tst.shape)


# 빈도가 낮은 타겟은 사전에 제거 (이유: 교차 검증에 활용할 수 없음 + 너무 빈도가 낮아 무의미함)
rem_targets = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 21, 22, 23]  # 18 classes
trn = trn[target['target'].isin(rem_targets)]
target = target[target['target'].isin(rem_targets)]
target = LabelEncoder().fit_transform(target)

for t in np.unique(target):
    print(t, sum(target==t))


#class Ensemble_sklearn:
	# Ensemble Method
	# RandomForest,

def evaluate(x, y, model):
    trn_scores = dict(); vld_scores = dict()
    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.1, random_state=777)
    for t_ind, v_ind in sss.split(x,y):
        # split data
        x_trn, x_vld = x.iloc[t_ind], x.iloc[v_ind]
        y_trn, y_vld = y[t_ind], y[v_ind]

# fit model
        model.fit(x_trn, y_trn)

# eval _ trn        
        preds = model.predict_proba(x_trn)

        log_scores = trn_scores.get('log loss', [])
        log_scores.append(log_loss(y_trn, preds))
        trn_scores['log loss'] = log_scores

# eval _ vld
        preds = model.predict_proba(x_vld)

        log_scores = vld_scores.get('log loss', [])
        log_scores.append(log_loss(y_vld, preds))
        vld_scores['log loss'] = log_scores
    return trn_scores, vld_scores

def print_scores_ensemble(trn_scores, vld_scores):
    prefix = '        '
    cols = ['log loss']
    print('='*50)
    print('TRAIN EVAL')
    for col in cols:
        print('-'*50)
        print('# {}'.format(col))
        print('# {} Mean : {}'.format(prefix, np.mean(trn_scores[col])))
        print('# {} Raw  : {}'.format(prefix, trn_scores[col]))

    print('='*50)
    print('VALID EVAL')
    for col in cols:
        print('-'*50)
        print('# {}'.format(col))
        print('# {} Mean : {}'.format(prefix, np.mean(vld_scores[col])))
        print('# {} Raw  : {}'.format(prefix, vld_scores[col]))

def print_time(end, start):
    print('='*50)
    elapsed = end - start
    print('{} secs'.format(round(elapsed)))
    
def fit_and_eval(trn, target, model):
    trn_scores, vld_scores = evaluate(trn,target,model)
    print_scores_with_logs(trn_scores, vld_scores)
    print_time(time.time(), st)    



# Utility

def observe_model_tree(trn, model):
    print('='*50)
    print(model)
    
    print('='*50)
    print('# Feature Importance')
    print(model.feature_importances_)
    
    print('-'*50)
    print('# Mapped to Column Name')
    prefix = '    '
    feature_importance = dict()
    for i, f_imp in enumerate(model.feature_importances_):
        print('{} {} \t {}'.format(prefix, round(f_imp,5), trn.columns[i]))
        feature_importance[trn.columns[i]] = f_imp

    print('-'*50)
    print('# Sorted Feature Importance')
    feature_importance_sorted = sorted(feature_importance.items(), key=operator.itemgetter(1), reverse=True)
    for item in feature_importance_sorted:
        print('{} {} \t {}'.format(prefix, round(item[1],5), item[0]))
    
    return feature_importance_sorted

def plot_fimp(fimp):
    x = []; y = []
    for item in fimp:
        x.append(item[0])
        y.append(item[1])

    f, ax = plt.subplots(figsize=(20, 15))
    sns.barplot(x,y,alpha=0.5)
    ax.set_title('Feature Importance for Model : Decision Tree')
    ax.set(xlabel='Column Name', ylabel='Feature Importance')



SOLUTION_NUM = "test"#"4"


import xgboost as xgb
# XGB Model Param
num_round = 5
early_stop = 10
xgb_params = {
    'booster': 'gbtree',
    
    # 모델 복잡도
    'max_depth': 5, # 높을 수록 복잡
    'gamma': 3,    # 낮을 수록 복잡
    'min_child_weight': 2, # 낮을 수록 복잡

    # 랜덤 샘플링을 통한 정규화
    'colsample_bylevel': 0.7,
    'colsample_bytree': 1,
    'subsample': 0.8,

    # 정규화
    'reg_alpha': 2,
    'reg_lambda': 3,

    # 학습 속도
    'learning_rate': 0.02,
    
    # 기본 설정
    'nthread': 4,
    'num_class': 18,
    'objective': 'multi:softprob',
    'silent': 1,
    'eval_metric': 'mlogloss',
    'seed': 777,
}

f = open("sol_"+SOLUTION_NUM+"_log.result","a")
f.write('num_round : '+str(num_round)+ '\n' + 'early_stop : '+str(early_stop)+'\n')
f.write(str(xgb_params))
f.close()


def evaluate_xgb(x, y):
    trn_scores = dict(); vld_scores = dict()
    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.1, random_state=777)
    for t_ind, v_ind in sss.split(x,y):
        # split data
        x_trn, x_vld = x.iloc[t_ind], x.iloc[v_ind]
        y_trn, y_vld = y[t_ind], y[v_ind]

        dtrn = xgb.DMatrix(x_trn, label=y_trn)
        dvld = xgb.DMatrix(x_vld, label=y_vld)
        watch_list = [(dtrn, 'train'), (dvld, 'eval')]

        # fit xgb
        bst = xgb.train(xgb_params, dtrn, num_round, watch_list, \
                        early_stopping_rounds=early_stop, verbose_eval=True)
        
        # eval _ trn        
        preds = bst.predict(dtrn)

        log_scores = trn_scores.get('log loss', [])
        log_scores.append(log_loss(y_trn, preds))
        trn_scores['log loss'] = log_scores

        # eval _ vld
        preds = bst.predict(dvld)
        
        log_scores = vld_scores.get('log loss', [])
        log_scores.append(log_loss(y_vld, preds))
        vld_scores['log loss'] = log_scores
    return trn_scores, vld_scores


def print_scores_with_logs(trn_scores, vld_scores):
    prefix = '        '
    cols = ['log loss']

    f = open("sol_"+SOLUTION_NUM+"_log.result","a")

    print('='*50)
    print('TRAIN EVAL')

    f.write('='*50)
    f.write('TRAIN EVAL')


    for col in cols:
        print('-'*50)
        print('# {}'.format(col))
        print('# {} Mean : {}'.format(prefix, np.mean(trn_scores[col])))
        print('# {} Raw  : {}'.format(prefix, trn_scores[col]))
	
        f.write('-'*50)
        f.write('# {}'.format(col))
        f.write('# {} Mean : {}'.format(prefix, np.mean(trn_scores[col])))
        f.write('# {} Raw  : {}'.format(prefix, trn_scores[col]))


    print('='*50)
    print('VALID EVAL')

    f.write('='*50)
    f.write('VALID EVAL')
 
    for col in cols:
        print('-'*50)
        print('# {}'.format(col))
        print('# {} Mean : {}'.format(prefix, np.mean(vld_scores[col])))
        print('# {} Raw  : {}'.format(prefix, vld_scores[col]))

        f.write('-'*50)
        f.write('# {}'.format(col))
        f.write('# {} Mean : {}'.format(prefix, np.mean(vld_scores[col])))
        f.write('# {} Raw  : {}'.format(prefix, vld_scores[col]))
    f.close()


def print_time(end, start):
    f = open("sol_"+SOLUTION_NUM+"_log.result","a")

    print('='*50)
    f.write('='*50) 
    elapsed = end - start
    print('{} secs'.format(round(elapsed)))
    f.write('{} secs'.format(round(elapsed)))
    f.close()
   
def fit_and_eval(trn, target, model):
    trn_scores, vld_scores = evaluate(trn,target,model)
    print_scores_with_logs(trn_scores, vld_scores)
    print_time(time.time(), st)    

evaluate_xgb(trn,target)



# XGBoost 기반 결과물 생성 코드
from datetime import datetime
import os

print('='*50)
print('# Test shape : {}'.format(tst.shape))

# 최종 모델 정의 및 학습 실행
dtrn = xgb.DMatrix(trn, label= target)
num_round = num_round # 평가 함수 기반 최적의 num_round 수치 지정
bst = xgb.train(xgb_params, dtrn, num_round, verbose_eval=True)

dtst = xgb.DMatrix(tst)
preds = bst.predict(dtst)
preds = np.fliplr(np.argsort(preds, axis=1))

cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
        'ind_cder_fin_ult1', 'ind_cno_fin_ult1',  'ind_ctju_fin_ult1',
        'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
        'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
        'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
        'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
        'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
        'ind_nomina_ult1',   'ind_nom_pens_ult1', 'ind_recibo_ult1']
target_cols = [cols[i] for i, col in enumerate(cols) if i in rem_targets]

final_preds = []
for pred in preds:
    top_products = []
    for i, product in enumerate(pred):
        top_products.append(target_cols[product])
        if i == 6:
            break
    final_preds.append(' '.join(top_products))

temp = pd.read_csv('../input/test_clean.csv')
test_id = temp['ncodpers']
out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})
file_name = datetime.now().strftime("result_"+SOLUTION_NUM+"_%Y%m%d%H%M%S") + '.csv'
out_df.to_csv(os.path.join('../output',file_name), index=False)


