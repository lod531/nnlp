analyzer=./src/analyzer.py
if [[ $1 = "1x1" ]]; then
    taskset --cpu-list 1 python3 $analyzer ./test_neural_nets/1x1.net ./../mnist_images/img0.txt 0.00001
elif [[ $1 = "2_2" ]]; then
    taskset --cpu-list 1 python3 $analyzer ./test_neural_nets/2x2.net ./../mnist_images/img0.txt 0.0000
elif [[ $1 = "3_50" ]]; then
    taskset --cpu-list 1 python3 $analyzer ./mnist_nets/mnist_relu_3_50.txt ./mnist_images/img1.txt 0.01
elif [[ $1 = "3_10" ]]; then
    taskset --cpu-list 1 python3 $analyzer ./mnist_nets/mnist_relu_3_10.txt ./mnist_images/img1.txt 0.1

elif [[ $1 = "9_100" ]]; then
    taskset --cpu-list 1 python3 $analyzer ./mnist_nets/mnist_relu_9_100.txt ./../mnist_images/img42.txt 0.025817
elif [[ $1 = "9_100s" ]]; then
    taskset --cpu-list 1 python3 $analyzer ./mnist_nets/mnist_relu_9_100.txt ./../mnist_images/img42.txt 0.01

elif [[ $1 = "9_200" ]]; then
    taskset --cpu-list 1 python3 $analyzer ./mnist_nets/mnist_relu_9_200.txt ./../mnist_images/img1.txt 0.01514
elif [[ $1 = "9_200x" ]]; then
    taskset --cpu-list 1 python3 $analyzer ./mnist_nets/mnist_relu_9_200.txt ./../mnist_images/img1.txt 0.02

elif [[ $1 = "vcomplex" ]]; then
    taskset --cpu-list 1 python3 $analyzer ./mnist_nets/mnist_relu_9_200.txt ./../mnist_images/img1.txt 0.01514
fi
