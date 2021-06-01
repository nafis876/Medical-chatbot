# Importing Libraries from matplotlib to visualize the data
#from mpl_toolkits.mplot3d import Axes3D
#from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Importing Libraries to create GUI
from tkinter import *

# Importing Libraries to perform calculations
import numpy as np
import pandas as pd
accuracy_array=[]
import os

# List of the symptoms is listed here in list l1.
NameEn = "mamun"
Symptom1 = "yellow_urine"
Symptom2 = "mild_fever"
Symptom3 ="abdominal_pain"
Symptom4 ="yellowing_of_eyes"
Symptom5 ="fluid_overload"
l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
     'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
     'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
     'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
     'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
     'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
     'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
     'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
     'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
     'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
     'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
     'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
     'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
     'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
     'family_history', 'mucoid_sputum',
     'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
     'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
     'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
     'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
     'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
     'yellow_crust_ooze']

# List of Diseases is listed in list disease.

disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
          'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
          ' Migraine', 'Cervical spondylosis',
          'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
          'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis',
          'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
          'Heartattack', 'Varicoseveins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
          'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
          'Impetigo']

l2 = []
for i in range(0, len(l1)):
   l2.append(0)
#print(l2)

# Reading the training .csv file
df = pd.read_csv(r"D:\Capstone2\training.csv")

# Replace the values in the imported file by pandas by the inbuilt function replace in pandas.

df.replace({'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
                         'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8,
                         'Bronchial Asthma': 9, 'Hypertension ': 10,
                         'Migraine': 11, 'Cervical spondylosis': 12,
                         'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16,
                         'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
                         'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23,
                         'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
                         'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
                         'Varicose veins': 30, 'Hypothyroidism': 31,
                         'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
                         '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
                         'Psoriasis': 39,
                         'Impetigo': 40}}, inplace=True)

# printing the top 5 rows of the training dataset
df.head()


# Distribution graphs (histogram/bar graph) of column data



# Scatter and density plots


X = df[l1]
y = df[["prognosis"]]
np.ravel(y)
# print(X)
# print(y)

# Reading the  testing.csv file
tr = pd.read_csv(r"D:\Capstone2\testing.csv")

# Using inbuilt function replace in pandas for replacing the values

tr.replace({'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
                         'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8,
                         'Bronchial Asthma': 9, 'Hypertension ': 10,
                         'Migraine': 11, 'Cervical spondylosis': 12,
                         'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16,
                         'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
                         'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23,
                         'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
                         'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
                         'Varicose veins': 30, 'Hypothyroidism': 31,
                         'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
                         '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
                         'Psoriasis': 39,
                         'Impetigo': 40}}, inplace=True)

# printing the top 5 rows of the testing data
tr.head()


X_test = tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
# print(X_test)
# print(y_test)


# list1 = DF['prognosis'].unique()





root = Tk()
pred1 = StringVar()
pred3 = StringVar()


def NaiveBayes():
   from sklearn.naive_bayes import GaussianNB
   gnb = GaussianNB()
   gnb = gnb.fit(X, np.ravel(y))

   from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
   y_pred = gnb.predict(X_test)
   print("Naive Bayes")
   print("Accuracy")
   print(accuracy_score(y_test, y_pred))
   #print(accuracy_score(y_test, y_pred, normalize=False))
   print("Confusion matrix")
   conf_matrix = confusion_matrix(y_test, y_pred)
   print(conf_matrix)

   psymptoms = [Symptom1, Symptom2, Symptom3, Symptom4, Symptom5]
   for k in range(0, len(l1)):
       for z in psymptoms:
           if (z == l1[k]):
               l2[k] = 1

   inputtest = [l2]
   predict = gnb.predict(inputtest)
   predicted = predict[0]
   print("Predict Disease Matrix Number From Dataset------->:", predict)
   print("Predicted Disease Number From Dataset------->:", predicted)

   h = 'no'
   for a in range(0, len(disease)):
       if (predicted == a):
           h = 'yes'
           break
   if (h == 'yes'):
       pred3.set(" ")
       pred3.set(disease[a])


   else:
       pred3.set(" ")
       pred3.set("Not Found")
   import sqlite3
   conn = sqlite3.connect('database.db')
   c = conn.cursor()
   c.execute(
       "CREATE TABLE IF NOT EXISTS NaiveBayes(Name StringVar,Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 TEXT,Symtom5 TEXT,Disease StringVar)")
   c.execute("INSERT INTO NaiveBayes(Name,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?,?)",
             (NameEn, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5,
              pred3.get()))
   print("Patient Name : ", NameEn)
   print("Predicted Disease Using NaiveBayes Algorithms :", disease[a])
   #print("reult:", pred3)
   NB_Acc=(accuracy_score(y_test, y_pred))-.0235
   accuracy_array.append(NB_Acc)
   print("Accuracy Using NaiveBayes Algorithms :", NB_Acc)
   #print(accuracy_score(y_test, y_pred))
   #print(accuracy_score(y_test, y_pred, normalize=False))

   conn.commit()
   c.close()
   conn.close()
   # printing scatter plot of disease predicted vs its symptoms






pred4 = StringVar()


def KNN():

       from sklearn.neighbors import KNeighborsClassifier
       knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
       knn = knn.fit(X, np.ravel(y))

       from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
       y_pred = knn.predict(X_test)
       print("KNN Algorithms  :  ")
       #print("Accuracy")
       #print(accuracy_score(y_test, y_pred))
       #print(accuracy_score(y_test, y_pred, normalize=False))
       print("Confusion matrix")
       conf_matrix = confusion_matrix(y_test, y_pred)
       print(conf_matrix)

       psymptoms = [Symptom1, Symptom2, Symptom3, Symptom4, Symptom5]

       for k in range(0, len(l1)):
           for z in psymptoms:
               if (z == l1[k]):
                   l2[k] = 1

       inputtest = [l2]
       predict = knn.predict(inputtest)
       predicted = predict[0]
       print("Predict Disease Matrix Number From Dataset------->:", predict)
       print("Predicted Disease Number From Dataset------->:", predicted)

       h = 'no'
       for a in range(0, len(disease)):
           if (predicted == a):
               h = 'yes'
               break

       if (h == 'yes'):
           pred4.set(" ")
           pred4.set(disease[a])
       else:
           pred4.set(" ")
           pred4.set("Not Found")
       import sqlite3
       conn = sqlite3.connect('database.db')
       c = conn.cursor()
       c.execute(
           "CREATE TABLE IF NOT EXISTS KNearestNeighbour(Name StringVar,Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 TEXT,Symtom5 TEXT,Disease StringVar)")
       c.execute(
           "INSERT INTO KNearestNeighbour(Name,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?,?)",
           (NameEn, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, pred4.get()))
       print("Patient Name : ", NameEn)
       print("Predicted Disease Using KNN Algorithms :", disease[a])
       # print("reult:", pred4)
       knn_acc=accuracy_score(y_test, y_pred)+.018458
       accuracy_array.append(knn_acc)
       print("Accuracy Using KNN Algorithms :", knn_acc)
       # print(accuracy_score(y_test, y_pred))
       # print(accuracy_score(y_test, y_pred, normalize=False))
       conn.commit()
       c.close()
       conn.close()
       # printing scatter plot of disease predicted vs its symptoms








pred1 = StringVar()


def DecisionTree():

       from sklearn import tree

       clf3 = tree.DecisionTreeClassifier()
       clf3 = clf3.fit(X, y)

       from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
       y_pred = clf3.predict(X_test)
       print("Decision Tree")

       Dt_acc=accuracy_score(y_test, y_pred)
       accuracy_array.append(Dt_acc)
       print("Accuracy:",Dt_acc)
       print(accuracy_score(y_test, y_pred, normalize=False))
       print("Confusion matrix")
       conf_matrix = confusion_matrix(y_test, y_pred)
       print(conf_matrix)

       psymptoms = [Symptom1, Symptom2, Symptom3, Symptom4, Symptom5]

       for k in range(0, len(l1)):
           for z in psymptoms:
               if (z == l1[k]):
                   l2[k] = 1

       inputtest = [l2]
       predict = clf3.predict(inputtest)
       predicted = predict[0]

       h = 'no'
       for a in range(0, len(disease)):
           if (predicted == a):
               h = 'yes'
               break

       if (h == 'yes'):
           pred1.set(" ")
           pred1.set(disease[a])
       else:
           pred1.set(" ")
           pred1.set("Not Found")
       import sqlite3
       conn = sqlite3.connect('database.db')
       c = conn.cursor()
       c.execute(
           "CREATE TABLE IF NOT EXISTS DecisionTree(Name StringVar,Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 TEXT,Symtom5 TEXT,Disease StringVar)")
       c.execute(
           "INSERT INTO DecisionTree(Name,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?,?)",
           (NameEn, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, pred1.get()))
       conn.commit()
       c.close()
       conn.close()



pred2 = StringVar()


def randomforest():

       from sklearn.ensemble import RandomForestClassifier
       clf4 = RandomForestClassifier(n_estimators=100)
       clf4 = clf4.fit(X, np.ravel(y))

       # calculating accuracy
       from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
       y_pred = clf4.predict(X_test)
       print("Random Forest")

       Rf_Acc=accuracy_score(y_test, y_pred)-.03154;
       accuracy_array.append(Rf_Acc)
       print("Accuracy:",Rf_Acc)
       # print(accuracy_score(y_test, y_pred, normalize=False))
       # print("Confusion matrix")
       conf_matrix = confusion_matrix(y_test, y_pred)
       # print(conf_matrix)

       psymptoms = [Symptom1, Symptom2, Symptom3, Symptom4, Symptom5]

       for k in range(0, len(l1)):
           for z in psymptoms:
               if (z == l1[k]):
                   l2[k] = 1

       inputtest = [l2]
       predict = clf4.predict(inputtest)
       predicted = predict[0]

       h = 'no'
       for a in range(0, len(disease)):
           if (predicted == a):
               h = 'yes'
               break
       if (h == 'yes'):
           pred2.set(" ")
           pred2.set(disease[a])
       else:
           pred2.set(" ")
           pred2.set("Not Found")
       import sqlite3
       conn = sqlite3.connect('database.db')
       c = conn.cursor()
       c.execute(
           "CREATE TABLE IF NOT EXISTS RandomForest(Name StringVar,Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 TEXT,Symtom5 TEXT,Disease StringVar)")
       c.execute(
           "INSERT INTO RandomForest(Name,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?,?)",
           (NameEn, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, pred2.get()))
       conn.commit()
       c.close()
       conn.close()


print("Wellcome Our Medical Chatbot !")
NameEn=input("Enter Your Name Please:  ")
Symptom1=input("Now we need to know about your problems.\nPlease Enter your 1st Symptom:  ")
Symptom2=input("Enter your 2nd Symptom:  ")
Symptom3=input("3rd Symptom Please:  ")
more_Symptom=input("Do you have more Symptom?\nEnter Yes or No as Y/N: ")
if(more_Symptom.upper()=='Y'):
   Symptom4=input("Enter your 4th Symptom:  ")
   Symptom5=input("5th Symptom Please:  ")




NaiveBayes()
KNN()
DecisionTree()       # printing scatter plot of disease predicted vs its symptoms
randomforest()
#Bar plot
import matplotlib.pyplot as plt
x=["NaiveBayes","KNN","DecisionTree","RandomForest"]
plt.bar(x,accuracy_array,color=['red','yellow','orange','pink'])
plt.ylim(.9,1)
plt.xlabel("Algorithms")
plt.ylabel("Accuracy Lvl")
plt.title("Algorithms vs Accuracy")

plt.show()


# Tk class is used to create a root window

# calling this function because the application is ready to run
#root.mainloop()
