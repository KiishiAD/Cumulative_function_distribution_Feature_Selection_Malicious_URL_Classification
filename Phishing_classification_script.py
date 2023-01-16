

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from joblib import parallel_backend


# # Read in Dataset

# In[3]:


#Reading in dataset using pandas
phish_data = pd.read_csv('Phishing_Legitimate_full.csv')

phish_data.info()


# # Transform Data

# In[4]:


# Transforming the data from int64 and float64 to int32 and float32 in order to save memory
integer_columns = phish_data.select_dtypes('int64')
for c in integer_columns:
    phish_data[c] = phish_data[c].astype('int32')
    
float_columns = phish_data.select_dtypes('float64')
for c in float_columns:
    phish_data[c] = phish_data[c].astype('float32')
phish_data.info()


# In[6]:


phish_data.rename(columns={'CLASS_LABEL': 'Class'}, inplace=True)


# In[7]:



# Count the number of instances of each value in the "AbnormalFormAction" column
value_counts = phish_data["AbnormalFormAction"].value_counts()

# Plot the value counts as a bar chart
value_counts.plot(kind='bar')

# Add axis labels and a title
plt.xlabel("AbnormalFormAction")
plt.ylabel("Count")
plt.title("AbnormalFormAction Distribution")

# Show the plot
plt.show()


# In[7]:


value_counts = phish_data["Class"].value_counts()

# Plot the value counts as a bar chart
value_counts.plot(kind='bar')

# Add axis labels and a title
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Class Distribution")

# Show the plot
plt.show()


# In[41]:


value_counts = phish_data["AtSymbol"].value_counts()

# Plot the value counts as a bar chart
value_counts.plot(kind='bar')

# Add axis labels and a title
plt.xlabel("AtSymbol")
plt.ylabel("Count")
plt.title("AtSymbol Distribution")

# Show the plot
plt.show()


# In[9]:


value_counts = phish_data["TildeSymbol"].value_counts()

# Plot the value counts as a bar chart
value_counts.plot(kind='bar')

# Add axis labels and a title
plt.xlabel("TildeSymbol")
plt.ylabel("Count")
plt.title("TildeSymbol Distribution")

# Show the plot
plt.show()


# In[10]:


value_counts = phish_data["NoHttps"].value_counts()

# Plot the value counts as a bar chart
value_counts.plot(kind='bar')

# Add axis labels and a title
plt.xlabel("NoHttps")
plt.ylabel("Count")
plt.title("NoHttps Distribution")

# Show the plot
plt.show()


# In[11]:


value_counts = phish_data["RandomString"].value_counts()

# Plot the value counts as a bar chart
value_counts.plot(kind='bar')

# Add axis labels and a title
plt.xlabel("RandomString")
plt.ylabel("Count")
plt.title("RandomString Distribution")

# Show the plot
plt.show()


# In[12]:


value_counts = phish_data["IpAddress"].value_counts()

# Plot the value counts as a bar chart
value_counts.plot(kind='bar')

# Add axis labels and a title
plt.xlabel("IpAddress")
plt.ylabel("Count")
plt.title("IpAddress Distribution")

# Show the plot
plt.show()


# In[13]:


value_counts = phish_data["DomainInSubdomains"].value_counts()

# Plot the value counts as a bar chart
value_counts.plot(kind='bar')

# Add axis labels and a title
plt.xlabel("DomainInSubdomains")
plt.ylabel("Count")
plt.title("DomainInSubdomains Distribution")

# Show the plot
plt.show()


# In[14]:


value_counts = phish_data["DomainInPaths"].value_counts()


value_counts.plot(kind='bar')
plt.xlabel("DomainInPaths")
plt.ylabel("Count")
plt.title("DomainInPaths Distribution")

plt.show()


# In[15]:


value_counts = phish_data["HttpsInHostname"].value_counts()

value_counts.plot(kind='bar')
plt.xlabel("HttpsInHostname")
plt.ylabel("Count")
plt.title("HttpsInHostname Distribution")


plt.show()


# In[16]:


value_counts = phish_data["DoubleSlashInPath"].value_counts()

value_counts.plot(kind='bar')
plt.xlabel("DoubleSlashInPath")
plt.ylabel("Count")
plt.title("DoubleSlashInPath Distribution")


plt.show()


# In[17]:


value_counts = phish_data["EmbeddedBrandName"].value_counts()

value_counts.plot(kind='bar')
plt.xlabel("EmbeddedBrandName")
plt.ylabel("Count")
plt.title("EmbeddedBrandName Distribution")


plt.show()


# In[18]:


value_counts = phish_data["ExtFavicon"].value_counts()

value_counts.plot(kind='bar')
plt.xlabel("ExtFavicon")
plt.ylabel("Count")
plt.title("ExtFavicon Distribution")


plt.show()


# In[19]:


value_counts = phish_data["InsecureForms"].value_counts()

value_counts.plot(kind='bar')
plt.xlabel("InsecureForms")
plt.ylabel("Count")
plt.title("InsecureForms Distribution")


plt.show()


# In[20]:


value_counts = phish_data["RelativeFormAction"].value_counts()

value_counts.plot(kind='bar')
plt.xlabel("RelativeFormAction")
plt.ylabel("Count")
plt.title("RelativeFormAction Distribution")


plt.show()


# In[21]:


value_counts = phish_data["ExtFormAction"].value_counts()

value_counts.plot(kind='bar')
plt.xlabel("ExtFormAction")
plt.ylabel("Count")
plt.title("ExtFormAction Distribution")


plt.show()


# In[22]:


value_counts = phish_data["FrequentDomainNameMismatch"].value_counts()

value_counts.plot(kind='bar')
plt.xlabel("FrequentDomainNameMismatch")
plt.ylabel("Count")
plt.title("FrequentDomainNameMismatch Distribution")


plt.show()


# In[23]:


value_counts = phish_data["FakeLinkInStatusBar"].value_counts()

value_counts.plot(kind='bar')
plt.xlabel("FakeLinkInStatusBar")
plt.ylabel("Count")
plt.title("FakeLinkInStatusBar Distribution")


plt.show()


# In[24]:


value_counts = phish_data["RightClickDisabled"].value_counts()

value_counts.plot(kind='bar')
plt.xlabel("RightClickDisabled")
plt.ylabel("Count")
plt.title("RightClickDisabled Distribution")


plt.show()


# In[25]:


value_counts = phish_data["PopUpWindow"].value_counts()

value_counts.plot(kind='bar')
plt.xlabel("PopUpWindow")
plt.ylabel("Count")
plt.title("PopUpWindow Distribution")


plt.show()


# In[26]:


value_counts = phish_data["SubmitInfoToEmail"].value_counts()

value_counts.plot(kind='bar')
plt.xlabel("SubmitInfoToEmail")
plt.ylabel("Count")
plt.title("SubmitInfoToEmail Distribution")


plt.show()


# In[27]:


value_counts = phish_data["IframeOrFrame"].value_counts()

value_counts.plot(kind='bar')
plt.xlabel("IframeOrFrame")
plt.ylabel("Count")
plt.title("IframeOrFrame Distribution")


plt.show()


# In[28]:


value_counts = phish_data["MissingTitle"].value_counts()

value_counts.plot(kind='bar')
plt.xlabel("MissingTitle")
plt.ylabel("Count")
plt.title("MissingTitle Distribution")


plt.show()


# In[29]:


value_counts = phish_data["ImagesOnlyInForm"].value_counts()

value_counts.plot(kind='bar')
plt.xlabel("ImagesOnlyInForm")
plt.ylabel("Count")
plt.title("ImagesOnlyInForm Distribution")


plt.show()


# # Spearmans Correlation
# For further explaratory data analysis we can look into which features are linearly correlated to get a idea into the general characteristics of a legitimate and illegitiame website

# In[30]:


def corr_heatmap(phish_data, idx_s, idx_e):
    y = phish_data['Class']
    temp = phish_data.iloc[:, idx_s:idx_e]
    if 'id' in temp.columns:
        del temp['id']
    temp['Class'] = y
    sns.heatmap(temp.corr(), annot=True, fmt='.2f')
    plt.show()


# In[31]:


corr_heatmap(phish_data, 0, 10)


# In[33]:


corr_heatmap(phish_data, 10, 20)


# In[34]:


corr_heatmap(phish_data, 20, 30)


# In[36]:


corr_heatmap(phish_data, 30, 40)


# In[8]:


# Split data into features and target
X = phish_data.drop(columns=['Class','id'])
y = phish_data['Class']

# Define the ensemble method
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)

# fit the classifier
clf2.fit(X, y)

# Compute the cumulative distribution function gradient
cdf_gradient = np.gradient(np.cumsum(clf2.feature_importances_))

# Select features using the ensemble method
sfm = SelectFromModel(estimator=clf2, threshold=np.percentile(cdf_gradient, 50))
sfm.fit(X, y)
X_transform = sfm.transform(X)

# Print the list of selected features
selected_features = X.columns[sfm.get_support()]
print(selected_features)



importances = sfm.estimator_.feature_importances_[sfm.get_support()]
sorted_indices = np.argsort(-importances)
plt.barh(np.array(selected_features)[sorted_indices], importances[sorted_indices])
plt.title('Selected Features and Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# In[9]:



    def logisticreg(data, top_n):
        
        top_features = sorted(selected_features, key=lambda x: -len(x))[:top_n]
        X = data[top_features]
        y = data['Class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

        lr = LogisticRegression(max_iter=10000)
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        return precision, recall, f1, accuracy
    
list = []
for i in range(20,51,1):
        precision, recall, f1, accuracy = logisticreg(phish_data, i)
        print("Performance for Logistic Model with Top {} features is precision : {}, recall : {}, f1 score : {}, accuracy : {}".format(i, precision, recall, f1, accuracy))
        list.append([i, precision, recall, f1, accuracy])

df = pd.DataFrame(list, columns=['num_of_features','precision', 'recall', 'f1_score', 'accuracy'])
print(df)

sns.lineplot(x='num_of_features', y='precision', data=df, label='Precision Score')
sns.lineplot(x='num_of_features', y='recall', data=df, label='Recall Score')
sns.lineplot(x='num_of_features', y='f1_score', data=df, label='F1 Score')
sns.lineplot(x='num_of_features', y='accuracy', data=df, label='Acc Score')


# In[13]:


# Defined a function to train and test the random forest classifier model
def random_forest(data, top_n, n_estimators=100, max_depth=None):
    top_features = sorted(selected_features, key=lambda x: -len(x))[:top_n]
    X = phish_data[top_features]
    y = phish_data['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    return precision, recall, f1, accuracy

# Train and test the model with different number of features
list = []
for i in range(20, 51, 1):
    precision, recall, f1, accuracy = random_forest(phish_data, i)
    print("Performance for Random Forest with Top {} features is precision : {}, recall : {}, f1 score : {}, accuracy : {}".format(i, precision, recall, f1, accuracy))
    list.append([i, precision, recall, f1, accuracy])

# Create a dataframe to hold the results and visualize the performance
df = pd.DataFrame(list, columns=['num_of_features','precision', 'recall', 'f1_score', 'accuracy'])
print(df)

sns.lineplot(x='num_of_features', y='precision', data=df, label='Precision Score')
sns.lineplot(x='num_of_features', y='recall', data=df, label='Recall Score')
sns.lineplot(x='num_of_features', y='f1_score', data=df, label='F1 Score')
sns.lineplot(x='num_of_features', y='accuracy', data=df, label='Acc Score')


