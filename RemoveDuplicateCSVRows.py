import csv

def remove():
    file_in = 'Dataset.csv'
    file_out = 'Train.csv'
    with open(file_in, 'r') as fin, open(file_out, 'w') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        for row in reader:
         if any(field.strip() for field in row):
             writer.writerow(row)
             
        d = {}
        for row in reader:
            if all(col != '' for col in row):
                continue
            
            movieName = row[1]
            if movieName not in d:
                d[movieName] = row  
                writer.writerow(row)
    result = d.values()
        
    
if __name__ == '__main__':
    print("Please run the correct program! It's called run_predictor.py!")






