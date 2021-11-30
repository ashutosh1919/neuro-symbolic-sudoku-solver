from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1lzZTAbwA19XPg2Au-hDjShEuAGaVGjCP',
                                    dest_path='difflogic/envs/sudoku/sudoku.csv',
                                    overwrite=True)

gdd.download_file_from_google_drive(file_id='1fHl9ut18_PahxFm-LeOPeY0Kv3MsvFer',
                                    dest_path='models/sudoku.pth',
                                    overwrite=True)
