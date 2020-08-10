from subprocess import Popen, PIPE, STDOUT
import sys
import re
import random
import numpy as np
from statistics import mean

accuracy_for_seed_8_p = []
accuracy_for_seed_16_p = []
accuracy_for_seed_8_e = []
accuracy_for_seed_16_e = []


#random_seeds = random.sample(range(0, 100), 20)
random_seeds = np.linspace(0, 19, 20)

for i in random_seeds:
    i = int(i)
    print('On seed:', i)
    Popen(['python', 'fraction_xy.py', 'x_test.csv', 'y_test.csv', '0.08', str(i)], stdout=PIPE, stderr=STDOUT).wait()
    Popen(['python', 'rename_files.py', '8', str(i)], stdout=PIPE, stderr=STDOUT).wait()
    
    Popen(['python', 'fraction_xy.py', 'x_test.csv', 'y_test.csv', '0.16', str(i)], stdout=PIPE, stderr=STDOUT).wait()
    Popen(['python', 'rename_files.py', '16', str(i)], stdout=PIPE, stderr=STDOUT).wait()
    
    svm_process = Popen(['python', 'SVM-p8.py'], stdout=PIPE, stderr=STDOUT)
    svm_process.wait()
    for line in svm_process.stdout:
        line = line.decode("utf-8")
        if re.compile('0.\d+').match(line):
            accuracy_for_seed_8_p.append(float(line))
    
    svm_process = Popen(['python', 'SVM-p16.py'], stdout=PIPE, stderr=STDOUT)
    svm_process.wait()
    for line in svm_process.stdout:
        line = line.decode("utf-8")
        if re.compile('0.\d+').match(line):
            accuracy_for_seed_16_p.append(float(line))
            
    svm_process = Popen(['python', 'SVM-e8.py'], stdout=PIPE, stderr=STDOUT)
    svm_process.wait()
    for line in svm_process.stdout:
        line = line.decode("utf-8")
        if re.compile('0.\d+').match(line):
            accuracy_for_seed_8_e.append(float(line))
    
    svm_process = Popen(['python', 'SVM-e16.py'], stdout=PIPE, stderr=STDOUT)
    svm_process.wait()
    for line in svm_process.stdout:
        line = line.decode("utf-8")
        if re.compile('0.\d+').match(line):
            accuracy_for_seed_16_e.append(float(line))
    


for (accuracy_8, accuracy_16, random_seed) in zip(accuracy_for_seed_8_p, accuracy_for_seed_16_p, random_seeds):
    random_seed = int(random_seed)
    print('Accuracy for seed', str(random_seed) + ': (polynomial)')
    print('8:', accuracy_8)
    print('16:', accuracy_16)
    print('--------------------------------------')

print("\nPolynomial kernel average: ")
print("8: ", mean(accuracy_for_seed_8_p))
print("16: ", mean(accuracy_for_seed_16_p) , "\n")

for (accuracy_8, accuracy_16, random_seed) in zip(accuracy_for_seed_8_e, accuracy_for_seed_16_e, random_seeds):
    random_seed = int(random_seed)
    print('Accuracy for seed', str(random_seed) + ': (exponential)')
    print('8:', accuracy_8)
    print('16:', accuracy_16)
    print('--------------------------------------')
    
print("\nExponential kernel average: ")
print("8: ", mean(accuracy_for_seed_8_e))
print("16: ", mean(accuracy_for_seed_16_e))
