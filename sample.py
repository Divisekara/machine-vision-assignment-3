# import numpy as np

# # a = np.arange(6).reshape((3,2))
# # print('before tranpose -\n', a)
# # transposed_a = np.transpose(a)
# # print('after tranpose - \n', transposed_a)

# sample = np.array([1., 2., 3.])
# print(sample)

# sample.reshape((3,1))

# print(sample)


# import matplotlib.pyplot as plt
# import numpy as np

# x = np.linspace(0,20,100)
# plt.plot(x, np.sin(x))
# plt.show()


# L = [1,2,3,4,5,0]
# print(L.index(min(L)))

# import os
# path = "https://drive.google.com/drive/folders/1kHzTWqNNGR_Js4XzOUC0InZj86tJKX0r"
# files = os.listdir(path)

# print(files)



page_token = None
while True:
    response = drive_service.files().list(q="mimeType='image/jpeg'",
                                          spaces='drive',
                                          fields='nextPageToken, files(id, name)',
                                          pageToken=page_token).execute()
    for file in response.get('files', []):
        # Process change
        print ('Found file: %s (%s)' % (file.get('name'), file.get('id')))
    page_token = response.get('nextPageToken', None)
    if page_token is None:
        break



