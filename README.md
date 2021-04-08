# cz4034-information-retrieval-

1. Install packages in a separate venv
```
pip install -r requirements.txt
```
or conda env (preferred)
```
conda create --name <env> --file requirements.txt
```

2. Create a folder named `assets` and place the three models (`.bin` files) downloaded from the [OneDrive link](https://entuedu-my.sharepoint.com/personal/c180065_e_ntu_edu_sg/_layouts/15/onedrive.aspx?originalPath=aHR0cHM6Ly9lbnR1ZWR1LW15LnNoYXJlcG9pbnQuY29tLzpmOi9nL3BlcnNvbmFsL2MxODAwNjVfZV9udHVfZWR1X3NnL0VyY2ZkRXEwdVhWRG9sdFo3Q0FSTDY0QnUwaS1sVDEwNERSRi1FWHRfMlduTnc%5FcnRpbWU9RDBiS2w5bjAyRWc&id=%2Fpersonal%2Fc180065%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FParler%2FSentiment%20Analysis)

3. To run the app

```
streamlit run app.py
```