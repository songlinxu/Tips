# Arduino -- Python communication shows "Permission Error" and "Access is Denied"

- Check if you have opened the Serial monitor on Arduino IDE. If you have opened it, then close it. You could not open the Serial Monitor and communicate with python at the same time.

# Do not open the arduino-python communication while uploading your code on arduino IDE



# Serial Communication Issue

- Always check the exact output of serial: you could use Serial.print("\[")
- Use Serial.println for multiple times may results in unexpected output

