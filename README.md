# Machine-Man
This is the PredictionIO template used to keep track of and build the training data. This is a slight modification of the default Text Classification template, as instead of just using one "text":"..." field, we are storing multiple attributes into the PIO app and training our model from each such attribute.

There are four required attributes that must always be present in any packet of data sent to the database:
  1. "folder": The folder where the e-mail is stored in Outlook.
  2. "ret": The return address of the e-mail, if present in the header.
  3. "subject": The subject of the e-mail.
  4. "from": The e-mail's author.
  5. "label": The classification of the data as either "machine" or "human".

If any of these previously listed fields is missing under the list of attributes stored in the pio app, the engine won't be able to train the data. You are free to pass as many extra attributes as you please. In the default app running on aws, there will be a disparity in the number of attributes between data sets as they imported both from previous tinman json files and a live Outlook instance, and the prior has relatively lesser attributes when compared against the latter.

Important: When deploying the pio server, remember to use ```pio deploy -- --driver-memory 2G```
