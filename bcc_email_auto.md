- Open the Script Editor. You can find it using Spotlight Search (Cmd + Space) and typing Script Editor.
- Create a New Script and input codes below.
- Run the Script

```
set filePath to (POSIX path of (path to desktop)) & "recipients.csv"

set emailBCC to ""
set emailSubject to "Your Email Subject"
set emailBody to "Your email body here."

-- Read the CSV file
set fileRef to open for access file filePath
set fileContent to read fileRef
close access fileRef

set emailList to paragraphs of fileContent

repeat with i from 2 to count of emailList -- Skip the header row
    set emailAddress to item i of emailList
    if emailAddress contains "@" then
        set emailBCC to emailBCC & emailAddress & ","
    end if
end repeat

-- Remove the last comma
if length of emailBCC > 0 then
    set emailBCC to text 1 thru -2 of emailBCC
end if

-- Create and send the email
tell application "Microsoft Outlook"
    set newMessage to make new outgoing message with properties {subject:emailSubject, content:emailBody}
    tell newMessage
        make new recipient with properties {email address:{address:emailBCC}}
        send
    end tell
end tell

```