import main
for type in ["easy","medium","hard"]:
    average_time=0
    for i in range(0,10):
        file=type + str(i) + ".jpg"
        print(file)
        #try:
        x=main.main_test(file, do_output_drawing=False, do_solving=True, show_detected_board=False,main_enable_debug=False)
        if x==None:
            x=0
        average_time+=x
    #print("Average time for " + type + " class is" + str(average_time / 10))