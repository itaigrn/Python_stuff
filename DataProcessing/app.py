import pandas as pd
import numpy as np
import sqlite3

from flask import Flask, request, render_template, jsonify
import os



zeta_counter=1

def read_file(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        df = df.dropna(how='all',axis=0)
        df = df.dropna(how='all',axis=1)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
        df = df.dropna(how='all',axis=0)
        df = df.dropna(how='all',axis=1)
    else:
        raise ValueError("Unsupported file format. Only CSV or Excel files are acceptable.")
    return df

def process_tns(df):
    df_sum = df.iloc[1:, 1:]
    reshaped = df_sum.to_numpy().reshape(df_sum.shape[0], -1, 3)
    summed_cols = reshaped.sum(axis=2)/3 # averaged out
    my_df = summed_cols[:,:-1]/summed_cols[:,-1:]
    my_df = my_df.reshape(24,)


    output = []
    if(~(my_df <= 10).any().any()):

        conn = sqlite3.connect('my_tns.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS formulations (
            formulation_id TEXT,
            calculated_value REAL
        )
        ''')

        #preparing data for insertion
        ids = []
        for i in range(len(my_df)):
            global zeta_counter
            ids.append(str(zeta_counter))
            zeta_counter+=1

        ids = pd.Series(ids)
        pandas_series = pd.Series(my_df)
        result = pd.DataFrame({'formulation_id': ids, 'calculated_value': pandas_series})
        data = result.to_records(index=False).tolist()

        #insertion
        cursor.executemany('''
        INSERT INTO formulations (formulation_id, calculated_value)
        VALUES (?, ?)
        ''', data)
        conn.commit()
        cursor.close()
        conn.close()

    else:
        formulations = np.where(my_df <= 10)
        formulations = formulations[0].tolist()
        for i in range(len(formulations)):
            #output.append(int(1+rows[i]+cols[i]*3)) ## not sorted
            output.append(1+int(formulations[i]))

    return output

def process_zeta(df):
    df_sum = df.iloc[:, 2:]
    reshaped = df_sum.to_numpy().reshape(-1, 3)
    summed_cols = reshaped.sum(axis=1)/3 # averaged out
    my_df = summed_cols[1:]/summed_cols[0]

    output = []
    if (~(my_df <= 0).any().any()):

        conn = sqlite3.connect('my_zeta.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS formulations (
            formulation_id TEXT,
            calculated_value REAL
        )
        ''')

        #data preparing for insertion
        idx = [3 * (i + 1) for i in range(df.shape[0] // 3 - 1)]
        new_df = df.iloc[idx, 1]
        new_df = new_df.reset_index(drop=True)
        pandas_series = pd.Series(my_df)
        result = pd.DataFrame({'formulation_id': new_df, 'calculated_value': pandas_series})
        data = result.to_records(index=False).tolist()

        #insertion
        cursor.executemany('''
        INSERT INTO formulations (formulation_id, calculated_value)
        VALUES (?, ?)
        ''', data)
        conn.commit()
        cursor.close()
        conn.close()

    else:
        form = np.where(my_df <= 0)
        output = [3*(1+int(fn)) for fn in form[0].tolist()]
        output = df.iloc[:, 1][output].tolist()

    return output


def the_stats(instrument):
    if instrument == 'zeta':
        my_doc = 'my_zeta.db'
    elif instrument == 'tns':
        my_doc = 'my_tns.db'
    else:
        raise ValueError("Unsupported experiment format. Only 'zeta' or 'tns' are supported.")

    conn = sqlite3.connect(my_doc)
    cursor = conn.cursor()

    # if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='formulations'")
    table_exists = cursor.fetchone()

    if not table_exists:
        conn.close()
        raise ValueError("Table 'formulations' does not exist.")

    # average
    cursor.execute('SELECT AVG(calculated_value) FROM formulations')
    average = cursor.fetchone()[0]

    if average is None:
        conn.close()
        raise ValueError("No data in 'formulations' table.")

    cursor.execute('SELECT calculated_value FROM formulations')
    values = [row[0] for row in cursor.fetchall()]

    if not values:
        conn.close()
        raise ValueError("No data available to calculate statistics.")

    # variance and standard deviation
    variance = np.var(values)
    std = float(np.sqrt(variance))

    # median
    df = pd.DataFrame(values, columns=['calculated_value'])
    median = float(df['calculated_value'].median())

    conn.close()
    return (median, average, std)

## web

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def process_file():
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        df = read_file(file_path)
        if df.shape == (9, 13) and ~df.isna().any().any():
            output = process_tns(df)
        elif df.shape[1] == 3 and  ~df.isna().any().any():
            output = process_zeta(df)
        else:
            raise ValueError("Unsupported experiment format.")
        return jsonify({'status': 'success', 'output': output})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/stats/<instrument>')
def get_statistics(instrument):
    try:
        stats = get_stats(instrument)
        return jsonify({'status': 'success', 'stats': stats})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/show_db')
def show_db():
    db_type = request.args.get('type')
    db_name = 'my_tns.db' if db_type == 'tns' else 'my_zeta.db'

    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM formulations")
        data = cursor.fetchall()
        conn.close()

        return jsonify({
            'status': 'success',
            'data': [{'formulation_id': row[0], 'calculated_value': row[1]} for row in data]
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_stats')
def get_stats():
    db_type = request.args.get('type')
    try:
        median, average, std = the_stats(db_type)
        return jsonify({
            'status': 'success',
            'median': median,
            'average': average,
            'std': std
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
