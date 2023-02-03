# Arduino -- Python communication shows "Permission Error" and "Access is Denied"

- Check if you have opened the Serial monitor on Arduino IDE. If you have opened it, then close it. You could not open the Serial Monitor and communicate with python at the same time.

# Do not open the arduino-python communication while uploading your code on arduino IDE



# Serial Communication Issue

- Always check the exact output of serial: you could use Serial.print("\["); Serial.print(receivedCommand); Serial.print("\]\n"). Then you will realize that the received value: receivedCommand is not a simple string. Instead, it is sth like "\nreceivedCommand\r\n" You need to remove the head and tail. 
- Use Serial.println for multiple times may results in unexpected output


