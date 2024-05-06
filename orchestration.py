import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
from prefect import task,flow



@task
def DataIngestion(file_path):
    return pd.read_csv(file_path,encoding="utf-8")

@task
def splitting_input_output(data,inputs,outputs):
    x=data[inputs]
    y=data[outputs]
    return x,y

@task
def splitting_data_into_train_test(x,y,test_size=0.2,random_state=43):
    return train_test_split(x,y,test_size=test_size,random_state=random_state)

@task
def Data_preprocessing(x_train,x_test,y_train,y_test):
    scaler=StandardScaler()
    x_train_scaled=scaler.fit_transform(x_train)
    x_test_scaled=scaler.transform(x_test)
    return x_train_scaled,x_test_scaled,y_train,y_test

@task
def model_training(x_train_scaled,y_train,hyperparameters):
    clf=RandomForestClassifier(**hyperparameters)
    clf.fit(x_train_scaled,y_train)
    return clf

@task
def model_evaluation(model,x_train_scaled,y_train,x_test_scaled,y_test):
    y_train_pred=model.predict(x_train_scaled)
    y_test_pred=model.predict(x_test_scaled)
    train_score=accuracy_score(y_train,y_train_pred)
    test_score=accuracy_score(y_test,y_test_pred)
    return train_score,test_score



@flow(name="Random forest training flow")
def workflow():
    path="final_data.csv"
    inputs=["ReserveStatus","Male","Price","CouponDiscount","Domestic","TripReason","Vehicle","From_encoded","to_encoded","Days_Difference","Family_Members","Ticket_Count"]
    outputs="Cancel"
    hyperparameters={"n_estimators":200}
    hotel=DataIngestion(path)
    x,y=splitting_input_output(hotel,inputs,outputs)
    x_train,x_test,y_train,y_test=splitting_data_into_train_test(x,y)
    x_train_scaled,x_test_scaled,y_train,y_test=Data_preprocessing(x_train,x_test,y_train,y_test)
    model=model_training(x_train_scaled,y_train,hyperparameters)
    train_score,test_score=model_evaluation(model,x_train_scaled,y_train,x_test_scaled,y_test)
    print("train score:",train_score)
    print("test score:",test_score)



if __name__=="__main__":
    workflow.serve(name="hotel_cancellation_prediction_production",
                   cron="* * * * *")

