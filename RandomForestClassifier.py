
##################################################
# Import libraries for data processing/evaluation
##################################################
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#########################################
# Import Random Forest Classifier from scikit-learn
#########################################
from sklearn.ensemble import RandomForestClassifier
######################################
# Import evaluation metrics libraries
######################################
from sklearn.metrics import confusion_matrix, classification_report

########################################
# Read .xlsx file from appropriate path
# Create DataFrame using Pandas
########################################
df = pd.read_excel("/Users/carolinesmith/mushrooms.xlsx")

#################################
# Display data & show attributes
#################################
print(df)
df.info()

############## PREPROCESSING ###############
# Transform labels to numerical
############################################
encoder = LabelEncoder()
for i in df.columns:
    df[i] = encoder.fit_transform(df[i])

#########################################################################
# Get data from X and Y axes - Features and Labels
# X - Drop "class" row, so it is not included in evaluation
# Y - Column "class"
#########################################################################
X = df.drop("class", axis = 1)
Y = df["class"]

#################################################
# Train-test-split
# Proportion 6/4 - 60% data used to train model
# 40% data used to test and evaluate performance
#################################################
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = 0.6, test_size = 0.4)

#######################################
# Declare variable 'rf_class' as model
# Train model using 'fit' method
#######################################
rf_class = RandomForestClassifier()
rf_class.fit(x_train, y_train)

############### PREDICTIONS ###############
# Predictions using trained model
###########################################
predictions = rf_class.predict(x_test)

# Write predictions, confusion matrix, classification report with F1 score and precision to text file
f = open("RFresults.txt", 'w')
f.write("Predictions:\n")
f.write(str(list(predictions)))
#f.write("Random Forest Score: ", rf_class.score(y_test))
f.write("\nConfusion Matrix:\n")
f.write(str(confusion_matrix(predictions, y_test)))
f.write("\nClassification Report:\n")
f.write(str(classification_report(y_test, predictions)))