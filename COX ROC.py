import pandas as pd
from lifelines import CoxPHFitter
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

optimization = ["FGOA"]
for opt in optimization:
    file = f"{opt}_all_sel_stage.csv"
    path = f'.\\FGOA\\stage'
    data = pd.read_csv(os.path.join(path, file))

    X = data.drop(['submitter_id'], axis=1)
    y = data['os']
    X_train_KNN, X_test_KNN, y_train_KNN, y_test_KNN = train_test_split(X,y, random_state=42, test_size=0.2)

    cph = CoxPHFitter()

    cph.fit(X_train_KNN, 'os_time', event_col='os', show_progress=True, robust=True)

    print("========")
    cph.print_summary()
    print("========")
    summary_df = cph.summary
    print("\n\nsummary:",summary_df,"\n\n")

    coefficients = cph.summary['coef']


    print("model coefficients:")
    print(coefficients)

    hazard_ratios = cph.summary['exp(coef)']
    p_value = cph.summary['p']


    print("\n\nhazard_ratios:")
    print(hazard_ratios)

    df = pd.DataFrame({
    'Feature': X.drop(['os', 'os_time'], axis=1).columns,
        })
    df = pd.concat([df, summary_df.reset_index(drop=True)], axis=1)

    df.to_csv(os.path.join(path, 'coxPH_Feature.csv'), index=False)

    # Plot hazard ratios
    ax = cph.plot(hazard_ratios=True)  # Plot without `fontsize`
    ax.tick_params(axis='both', which='major', labelsize=20)  # Set font size for tick labels
    ax.set_title("Hazard Ratios with 95% Confidence Intervals", fontsize=20)  # Set title font size
    ax.set_xlabel("Hazard Ratio", fontsize=20)  # Set x-axis font size
    ax.set_ylabel("Covariates", fontsize=20)
    plt.show()