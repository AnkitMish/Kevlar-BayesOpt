import subprocess


command=["squeue" " -u" "ankitmis " "|" " grep" " kev"]

result=subprocess.run(command, shell=True, stdout=subprocess.PIPE)
print(result.stdout)
print(len(result.stdout))
#        while (len(result.stdout) != 0):
#            result=subprocess.run(command, shell=True, stdout=subprocess.PIPE)
#            time.sleep(200)
#        os.chdir("..")

