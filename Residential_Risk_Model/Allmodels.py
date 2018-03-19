# The random forest model
model_rf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=27, max_features='log2')
model_rf.fit(select_X_train, y_train)
y_pred = model_rf.predict(select_X_test)
predictions = [round(value) for value in y_pred]
fpr, tpr, thresholds = metrics.roc_curve(impute_y_test, predictions, pos_label=1)
accuracy = accuracy_score(impute_y_test, predictions)
cm = confusion_matrix(impute_y_test, predictions)
print(confusion_matrix(impute_y_test, predictions))

kappa = cohen_kappa_score(impute_y_test, predictions)
acc = float(cm[0][0] + cm[1][1]) / len(impute_y_test)
auc = metrics.auc(fpr, tpr)
recall = tpr[1]
precis = float(cm[1][1]) / (cm[1][1] + cm[0][1])

print('Final Test Data Results')
print("Thresh=%d, n=%d" % (thres.iloc[0]['Thresh'], select_X_test.shape[1]))
print('Accuracy = {0} \n \n'.format(acc))
print('kappa score = {0} \n \n'.format(kappa))
print('AUC Score = {0} \n \n'.format(auc))
print('recall = {0} \n \n'.format(recall))
print('precision = {0} \n \n'.format(precis))


# The adaboost model
model_adaboost = AdaBoostClassifier(n_estimators=1000, random_state=27, algorithm='SAMME')
model_adaboost.fit(X_train, y_train)
#pred_adaboost = model_adaboost.predict(X_validation)
pred_adaboost = model_adaboost.predict(X_test)
#real_adaboost = y_validation
real_adaboost = y_test
cm_ada = confusion_matrix(real_adaboost, pred_adaboost)
print(cm_ada)

kappa_ada = cohen_kappa_score(real_adaboost, pred_adaboost)

# compute ROC curve and area under the curve
# fpr, tpr, thresholds = metrics.roc_curve(y_validation, pred_adaboost, pos_label=1)
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_adaboost, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

acc_ada = 'Accuracy = {0} \n \n'.format(float(cm_ada[0][0] + cm_ada[1][1]) / len(real_adaboost))
kapp_ada = 'kappa score = {0} \n \n'.format(kappa_ada)
auc_ada = 'AUC Score = {0} \n \n'.format(metrics.auc(fpr, tpr))
recall_ada = 'recall = {0} \n \n'.format(tpr[1])
precis_ada = 'precision = {0} \n \n'.format(float(cm_ada[1][1]) / (cm_ada[1][1] + cm_ada[0][1]))

print(acc_ada)
print(kapp_ada)
print(auc_ada)
print(recall_ada)
print(precis_ada)


# The XG Boost model
model_xgboost = XGBClassifier(learning_rate=0.13, n_estimators=1500,
                              objective='binary:logistic', nthread=4, seed=27)
model_xgboost.fit(X_train, y_train)
# pred_xgboost = model_xgboost.predict(X_validation)
pred_xgboost = model_xgboost.predict(X_test)
# real_xgboost = y_validation
real_xgboost = y_test
cm_xg = confusion_matrix(real_xgboost, pred_xgboost)
print(cm_xg)

from sklearn.metrics import cohen_kappa_score
kappa_xg = cohen_kappa_score(real_xgboost, pred_xgboost)

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_xgboost, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

acc_xg = 'Accuracy = {0} \n \n'.format(float(cm_xg[0][0] + cm_xg[1][1]) / len(real_xgboost))
kapp_xg = 'kappa score = {0} \n \n'.format(kappa_xg)
auc_xg = 'AUC Score = {0} \n \n'.format(metrics.auc(fpr, tpr))
recall_xg = 'recall = {0} \n \n'.format(tpr[1])
precis_xg = 'precision = {0} \n \n'.format(float(cm_xg[1][1]) / (cm_xg[1][1] + cm_xg[0][1]))

print(acc_xg)
print(kapp_xg)
print(auc_xg)
print(recall_xg)
print(precis_xg)
