import kivy
kivy.require('2.2.1')  # replace with your current kivy version !
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.graphics.texture import Texture

# Other dependencies
import cv2
from image_processing import get_grid_dimensions, filter_non_square_contours, sort_grid_contours, reduce_noise, transform_grid, get_cells_from_9_main_cells
from helper_functions import sudoku_cells_reduce_noise
import tensorflow as tf
from csp import csp, create_empty_board, BLANK_STATE
from backtracking import backtracking
import numpy as np
import copy
import imutils


class WindowManager(ScreenManager):
    pass


class MainMenuScreen(Screen):
    pass


class CameraScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None

        self.frame = None
        self.grid = None
        self.grid_coordinates = None
        self.cnts = None
        self.model = None
        self.solved = None

    def on_enter(self):
        # Load trained h5 model
        self.model = tf.keras.models.load_model('model.h5')

        self.capture = cv2.VideoCapture(0)
        # Loop
        Clock.schedule_interval(self.update, 1.0 / 33.0)

    def on_leave(self):
        self.capture.release()
        Clock.unschedule(self.update)

        # Clear all variables
        self.frame = None
        self.grid = None
        self.grid_coordinates = None
        self.cnts = None
        self.model = None
        self.solved = None

    def update(self, *args):
        # Read frame from opencv
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                # Resize frame
                # Check if image is too big
                # If so, Standardise image size to avoid error in cell image manipulation
                # Cells must fit in 28x28 for the model, big images will exceed this threshold with aspect ratio resize
                if frame.shape[1] > 700:
                    frame = imutils.resize(frame, width=700)

                # Convert frame to texture
                # Flip image horizontally, convert to str
                buf = cv2.flip(frame, 0).tostring()
                img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

                # Display texture, link to .kv id
                self.ids.frame_img.texture = img_texture

                # If detected grid, display solved grid as popup
                self.detect_grid(frame)

                if self.grid is not None and self.cnts is not None:
                    self.frame = frame

                    # Release handle to the webcam
                    self.capture.release()
                    Clock.unschedule(self.update)

                    # Detected valid grid
                    # Solve grid
                    self.solve_grid()

                    # If grid is solved, display solved grid
                    if self.solved is not None:
                        # Convert frame to texture
                        # Flip image horizontally, convert to str
                        buf = cv2.flip(self.solved, 0).tostring()
                        img_texture = Texture.create(size=(self.solved.shape[1], self.solved.shape[0]), colorfmt='bgr')
                        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

                        # Display texture, link to .kv id
                        self.ids.frame_img.texture = img_texture


    def detect_grid(self, frame):
        # Extract grid
        grid_coordinates = get_grid_dimensions(frame)

        # Check if grid is found
        if grid_coordinates is not None:
            # Crop grid with transformation
            grid = transform_grid(frame, grid_coordinates)
            self.grid = grid
            self.grid_coordinates = grid_coordinates

            # Image preprocessing, reduce noise such as numbers/dots, cover all numbers
            thresh = reduce_noise(grid)

            # Contour detection again, this time we are extracting the grid
            cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Filter out non square contours
            cnts = filter_non_square_contours(cnts)

            # Convert contours into data to work with
            # Check how many valid cnts are found
            if 9 <= (cnts_len := len(cnts)) <= 90:
                # Salvageable
                if cnts_len == 81:
                    # All cells extracted, perfect
                    self.cnts = cnts
                elif cnts_len == 9:
                    # Split main cells to 81 cells
                    cnts = get_cells_from_9_main_cells(cnts)
                    self.cnts = cnts
                else:
                    new_cnts = []

                    # In between, not sure if this is a valid grid
                    # Sort hierarchy, toss small contours to find main cells
                    # Only accept contours with hierarchy 0 (main contours)
                    # Format of hierarchy: [next, previous, child, parent]
                    for cnt, hie in zip(cnts, hierarchy[0]):
                        # Check if parent is -1 (Does not exist)
                        if hie[3] == -1:
                            new_cnts.append(cnt)

                    if len(new_cnts) == 9:
                        # Got all main cells
                        cnts = get_cells_from_9_main_cells(new_cnts)
                        self.cnts = cnts
                    else:
                        # Unable to identify main cells
                        print(f"Unable to extract grid cells properly from frame")

    def solve_grid(self):
        # Update contour len, in case any contour filtering/adjustment was made
        cnts_len = len(self.cnts)

        # Success detection of grid & cells
        # Sort grid into nested list format same as sudoku
        grid_contours = sort_grid_contours(self.cnts)

        # Create a blank Sudoku board
        board = create_empty_board()

        # Run digit classifier
        for row_index, row in enumerate(grid_contours):
            for box_index, box in enumerate(row):

                # Extract cell ROI from contour
                x, y, width, height = cv2.boundingRect(box)
                roi = self.grid[y:y + height, x:x + width]

                # Convert to greyscale
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Image thresholding & invert image
                digit_inv = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY_INV, 27, 11)

                # Remove surrounding noise
                digit = sudoku_cells_reduce_noise(digit_inv)

                # Digit present
                if digit is not None:
                    # Reshape to fit model input
                    digit = digit.reshape((1, 28, 28, 1))

                    # Make prediction
                    board[row_index][box_index] = np.argmax(self.model.predict(digit), axis=-1)[0] + 1

        # Perform backtracking/CSP to solve detected puzzle
        # If smaller amount of digits provided, use backtracking
        # Else CSP is faster
        if sum(cell.count(BLANK_STATE) for cell in board) > 70:
            # Backtracking, more than 70/81 blanks
            solved_board, steps = backtracking(copy.deepcopy(board))
        else:
            # CSP, less than 70/81 blanks
            solved_board, steps = csp(copy.deepcopy(board))

        # Check if puzzle is valid
        if steps:
            # Solved
            # Draw answers on the sudoku image
            for row_index, row in enumerate(board):
                for box_index, box in enumerate(row):
                    # Filter for BLANK_STATES
                    if box == BLANK_STATE:
                        x, y, width, height = cv2.boundingRect(grid_contours[row_index][box_index])

                        # Calculate font size
                        for num in np.arange(1.0, 10.0, 0.1):
                            text_size = cv2.getTextSize(str(solved_board[row_index][box_index]),
                                                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                                        fontScale=num, thickness=2)

                            font_size = num
                            if text_size[0][0] > width // 2 or text_size[0][1] > height // 2:
                                break

                        # Fill in answers in sudoku image
                        cv2.putText(self.frame, str(solved_board[row_index][box_index]),
                                    (x + self.grid_coordinates[0][0] + (width * 1 // 4),
                                     y + self.grid_coordinates[0][1] + (height * 3 // 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2)

            # Fill in information at bottom left
            cv2.putText(self.frame, f"Solved in {steps} steps",
                        (0, self.frame.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2)

            self.solved = self.frame

            print(f"Solved in {steps} steps")
        else:
            # Cannot be solved (Wrong/invalid puzzle)
            # Reasons can be invalid puzzle or grid/digits detected wrongly
            print(f"Invalid puzzle or digit detection error")
            self.solved = None


class GalleryScreen(Screen):
    pass


kv = Builder.load_file("sudokulens.kv")


class SudokuLensApp(App):
    def build(self):
        return kv


if __name__ == '__main__':
    SudokuLensApp().run()
