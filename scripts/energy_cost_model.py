from sklearn.linear_model import LinearRegression

### Task
#Use any applicable statistics/machine learning method to predict a person’s 
#expected energy expenditure for a given gradient.

### Proposal
#From the EDA, we can clearly see two differents linear dependence 
# for  gradient < 0 and for gradient > 0. 
# Therefore we can use a different linear regresor for each domain region.


def train_energy_cost_model(df_energy):
    # Separate the data into two DataFrames based on the sign of the gradient
    df_positive_gradient = df_energy[df_energy['gradient'] > 0]
    df_negative_gradient = df_energy[df_energy['gradient'] < 0]

    # Train a linear regression model for positive gradient region
    model_positive = LinearRegression()
    X_positive = df_positive_gradient[['gradient']]
    y_positive = df_positive_gradient['energy_cost']
    model_positive.fit(X_positive.values, y_positive.values)

    # Train a linear regression model for negative gradient region
    model_negative = LinearRegression()
    X_negative = df_negative_gradient[['gradient']]
    y_negative = df_negative_gradient['energy_cost']
    model_negative.fit(X_negative.values, y_negative.values)

    return model_positive, model_negative

def predict_energy_cost(altitude_diff, model_positive, model_negative):
    if altitude_diff > 0:
        return model_positive.predict([[altitude_diff]])[0]
    else:
        return model_negative.predict([[altitude_diff]])[0]
