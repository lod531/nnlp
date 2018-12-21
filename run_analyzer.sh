#taskset --cpu-list 1 python3 analyzer.py ./../mnist_nets/mnist_relu_3_10.txt ./../mnist_images/img0.txt 0.00001

#taskset --cpu-list 1 python3 analyzer.py ./../mnist_nets/mnist_relu_3_10.txt ./../mnist_images/img0.txt 0.00001

if [[ $1 = "1x1" ]]; then
    taskset --cpu-list 1 python3 analyzer.py ./test_neural_nets/1x1.net ./../mnist_images/img0.txt 0.00001
elif [[ $1 = "2_2" ]]; then
    taskset --cpu-list 1 python3 analyzer.py ./test_neural_nets/2x2.net ./../mnist_images/img0.txt 0.0000
elif [[ $1 = "6_100" ]]; then
    taskset --cpu-list 1 python3 analyzer.py ./../mnist_nets/mnist_relu_6_100.txt ./../mnist_images/img1.txt 0.005
elif [[ $1 = "3_50" ]]; then
    taskset --cpu-list 1 python3 analyzer.py ./../mnist_nets/mnist_relu_3_50.txt ./../mnist_images/img1.txt 0.01
elif [[ $1 = "3_10" ]]; then
    taskset --cpu-list 1 python3 analyzer.py ./../mnist_nets/mnist_relu_3_10.txt ./../mnist_images/img1.txt 0.4

elif [[ $1 = "6x200" ]]; then
    taskset --cpu-list 1 python3 analyzer.py ./../mnist_nets/mnist_relu_6_200.txt ./../mnist_images/img1.txt 0.01
elif [[ $1 = "9_100" ]]; then
    taskset --cpu-list 1 python3 analyzer.py ./../mnist_nets/mnist_relu_9_100.txt ./../mnist_images/img42.txt 0.025817
elif [[ $1 = "9_100s" ]]; then
    taskset --cpu-list 1 python3 analyzer.py ./../mnist_nets/mnist_relu_9_100.txt ./../mnist_images/img42.txt 0.01

elif [[ $1 = "9_200" ]]; then
    taskset --cpu-list 1 python3 analyzer.py ./../mnist_nets/mnist_relu_9_200.txt ./../mnist_images/img1.txt 0.01514
elif [[ $1 = "9_200x" ]]; then
    taskset --cpu-list 1 python3 analyzer.py ./../mnist_nets/mnist_relu_9_200.txt ./../mnist_images/img1.txt 0.02

elif [[ $1 = "vcomplex" ]]; then
    taskset --cpu-list 1 python3 analyzer.py ./../mnist_nets/mnist_relu_9_200.txt ./../mnist_images/img1.txt 0.01514
elif [[ $1 = "4_1024" ]]; then
    taskset --cpu-list 1 python3 analyzer.py ./../mnist_nets/mnist_relu_4_1024.txt ./../mnist_images/img1.txt 0.005

elif [[ $1 = "medium" ]]; then
    taskset --cpu-list 1 python3 analyzer.py ./../mnist_nets/mnist_relu_3_50.txt ./../mnist_images/img1.txt 0.035
elif [[ $1 = "testo" ]]; then
    img=./../mnist_images/img1.txt
  	for net in ./../mnist_nets/*.txt; do
        if [[ $net != "./../mnist_nets/mnist_relu_4_1024.txt" && $net != "./../mnist_nets/mnist_relu_200.txt" && $net != "./../mnist_nets/mnist_relu_9_200" ]]; then
            taskset --cpu-list 1 python3 stats_analyzer.py $net $img 0.01
        fi
    done	   
    img=./../mnist_images/img0.txt
  	for net in ./../mnist_nets/*.txt; do
        if [[ $net != "./../mnist_nets/mnist_relu_4_1024.txt" && $net != "./../mnist_nets/mnist_relu_200.txt" && $net != "./../mnist_nets/mnist_relu_9_200" ]]; then
            taskset --cpu-list 1 python3 stats_analyzer.py $net $img 0.01
        fi
    done 
    img=./../mnist_images/img2.txt
   	for net in ./../mnist_nets/*.txt; do
        if [[ $net != "./../mnist_nets/mnist_relu_4_1024.txt" && $net != "./../mnist_nets/mnist_relu_200.txt" && $net != "./../mnist_nets/mnist_relu_9_200" ]]; then
            taskset --cpu-list 1 python3 stats_analyzer.py $net $img 0.01
        fi
    done 
elif [[ $1 = "testo_test" ]]; then
    taskset --cpu-list 1 python3 stats_analyzer.py ./../mnist_nets/mnist_relu_6_50.txt ./../mnist_images/img1.txt 0.01
#    taskset --cpu-list 1 python3 stats_analyzer.py ./../mnist_nets/mnist_relu_6_100.txt ./../mnist_images/img1.txt 0.01
#    taskset --cpu-list 1 python3 stats_analyzer.py ./../mnist_nets/mnist_relu_6_200.txt ./../mnist_images/img1.txt 0.01


elif [[ $1 = "test" ]]; then
    STR="ASD"
    echo $STR
elif [[ $1 = "ground_truth" ]]; then
    echo "Ground truth:"
    netname="./../mnist_nets/mnist_relu_3_10.txt"
    echo "$netname"
	for filename in ./../mnist_images/img*; do
        taskset --cpu-list 1 python3 range_analyzer.py "$netname" "$filename" 0.01 >> deleteMe.txt
	done
    netname="./../mnist_nets/mnist_relu_3_20.txt"
    echo "$netname"
	for filename in ./../mnist_images/img*; do
        taskset --cpu-list 1 python3 range_analyzer.py "$netname" "$filename" 0.01 >> deleteMe.txt
	done
    netname="./../mnist_nets/mnist_relu_3_50.txt"
    echo "$netname"
	for filename in ./../mnist_images/img*; do
        taskset --cpu-list 1 python3 range_analyzer.py "$netname" "$filename" 0.01 >> deleteMe.txt
	done
    netname="./../mnist_nets/mnist_relu_6_20.txt"
    echo "$netname"
	for filename in ./../mnist_images/img*; do
        taskset --cpu-list 1 python3 range_analyzer.py "$netname" "$filename" 0.01 >> deleteMe.txt
	done
    netname="./../mnist_nets/mnist_relu_6_50.txt"
    echo "$netname"
	for filename in ./../mnist_images/img*; do
        taskset --cpu-list 1 python3 range_analyzer.py "$netname" "$filename" 0.01 >> deleteMe.txt
	done
    echo "Check deleteMe.txt"
elif [[ $1 = "verify" ]]; then
 	for filename in ./../mnist_images/*.txt; do
        echo "$filename" 
    done	   
fi
