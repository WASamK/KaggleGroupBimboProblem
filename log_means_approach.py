import pandas as pd
import numpy as np

# Reading Data
data_types_train = {'Agencia_ID': np.uint16, 'Ruta_SAK': np.uint16, 'Cliente_ID': np.uint32,
                    'Producto_ID': np.uint16, 'Demanda_uni_equil': np.uint32}
data_types_test = {'Agencia_ID': np.uint16, 'Ruta_SAK': np.uint16, 'Cliente_ID': np.uint32,
                   'Producto_ID': np.uint16}
train = pd.read_csv('train.csv', usecols=data_types_train.keys(), dtype=data_types_train)
test = pd.read_csv('test.csv', usecols=data_types_test.keys(), dtype=data_types_test)

# Converting the demand to obtain the RMSLE log(1 + demand)
train['Log_Demanda_uni_equil'] = 1.006999 * np.log1p(train['Demanda_uni_equil'] + 0.01159) - 0.01159
mean_overall = np.mean(train['Log_Demanda_uni_equil'])

# Computing the mean value by grouping by each attribute

# Grouping by product ID
Product_group_mean = pd.DataFrame(
    {'m_Product': train.groupby('Producto_ID')['Log_Demanda_uni_equil'].mean()}).reset_index()
# Grouping by client ID
Client_group_mean = pd.DataFrame(
    {'m_Client': train.groupby('Cliente_ID')['Log_Demanda_uni_equil'].mean()}).reset_index()
# Grouping by product ID and agency ID
Product_Agent_group_mean = pd.DataFrame(
    {'m_Product_Agent': train.groupby(['Producto_ID', 'Agencia_ID'])['Log_Demanda_uni_equil'].mean()}).reset_index()
# Grouping by product ID and route ID
Product_Route_group_mean = pd.DataFrame(
    {'m_Product_Route': train.groupby(['Producto_ID', 'Ruta_SAK'])['Log_Demanda_uni_equil'].mean()}).reset_index()
# Grouping by product ID, client ID and agency ID
Product_Client_Agent_group_mean = pd.DataFrame({'m_Product_Client_Agent':
                                                    train.groupby(['Producto_ID', 'Cliente_ID', 'Agencia_ID'])[
                                                        'Log_Demanda_uni_equil'].mean()}).reset_index()

# Merging the grouped mean values with the test data set

# Grouping by product ID
test = test.merge(Product_group_mean, how='left', on=["Producto_ID"])
# Grouping by client ID
test = test.merge(Client_group_mean, how='left', on=["Cliente_ID"])
# Grouping by product ID and agency ID
test = test.merge(Product_Agent_group_mean, how='left', on=["Producto_ID", "Agencia_ID"])
# Grouping by product ID and route ID
test = test.merge(Product_Route_group_mean, how='left', on=["Producto_ID", "Ruta_SAK"])
# Grouping by product ID, client ID and agency ID
test = test.merge(Product_Client_Agent_group_mean, how='left', on=["Producto_ID", "Cliente_ID", "Agencia_ID"])

# Computing the inverse of RMSLE function to obtain the exact values for the predicton
test['Demanda_uni_equil'] = np.expm1(test['m_Product_Client_Agent']) * 0.7173 + np.expm1(
    test['m_Product_Route']) * 0.1849 + 0.126

# Computing the missing values in the columns
test.loc[test['Demanda_uni_equil'].isnull(), 'Demanda_uni_equil'] = test.loc[test[
                                                                                 'Demanda_uni_equil'].isnull(), 'm_Product_Route'].apply(
    np.expm1) * 0.745 + 0.18
test.loc[test['Demanda_uni_equil'].isnull(), 'Demanda_uni_equil'] = test.loc[test[
                                                                                 'Demanda_uni_equil'].isnull(), 'm_Client'].apply(
    np.expm1) * 0.82 + 0.86
test.loc[test['Demanda_uni_equil'].isnull(), 'Demanda_uni_equil'] = test.loc[test[
                                                                                 'Demanda_uni_equil'].isnull(), 'm_Product_Agent'].apply(
    np.expm1) * 0.55 + 0.90
test.loc[test['Demanda_uni_equil'].isnull(), 'Demanda_uni_equil'] = test.loc[test[
                                                                                 'Demanda_uni_equil'].isnull(), 'm_Product'].apply(
    np.expm1) * 0.5 + 1
test.loc[test['Demanda_uni_equil'].isnull(), 'Demanda_uni_equil'] = np.expm1(mean_overall) - 0.9
test['Demanda_uni_equil'] = test['Demanda_uni_equil'].round(decimals=4)

# Saving the submission into csv
prediction = test[['id', 'Demanda_uni_equil']]
prediction.to_csv('log_means_submission.csv', index=False, columns=['id', 'Demanda_uni_equil'])
