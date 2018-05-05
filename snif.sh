sudo rmmod 8192cu
sudo modprobe rtl8192cu
sudo ip link set wlan1 down
sudo iw dev wlan1 set type monitor
sudo ip link set wlan1 up
sudo iw dev wlan1 set channel 1
