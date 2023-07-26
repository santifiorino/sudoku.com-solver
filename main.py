import cv2
import skimage
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pyautogui
from SudokuSolver import SudokuSolver

def main():
    # Make sure alt-tabbing switches to the browser where sudoku.com is open
    pyautogui.hotkey("alt", "tab", interval=0.1)
    # Take a screenshot of the screen and find the sudoku
    screenshot = pyautogui.screenshot()
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    preprocessed = preprocess(screenshot)
    square_contour = find_sudoku_contour(preprocessed)
    if square_contour is None:
        print("No sudoku found")
        return
    cropped_grid = crop_grid(screenshot, square_contour)
    # Split the sudoku into 81 squares and detect the digits
    squares_images = split_grid(cropped_grid)
    sudoku = squares_images_to_sudoku(squares_images)
    # Solve the sudoku
    solver = SudokuSolver(sudoku)
    solved = solver.solve()
    solve_on_website(square_contour, solved)

# -- Image preprocessing
def preprocess(screenshot):
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh

def find_sudoku_contour(preprocessed):
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for contour in contours:
        if is_square(contour):
            squares.append(contour)
    squares = sorted(squares, key=cv2.contourArea, reverse=True)
    if len(squares) == 0:
        return None
    return squares[0]

def is_square(contour):
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    _, _, w, h = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)
    return len(approx) == 4 and abs(aspect_ratio - 1) < 0.1

def crop_grid(screenshot, square):
    x, y, w, h = cv2.boundingRect(square)
    cropped = screenshot[y:y+h, x:x+w]
    return cropped

def split_grid(cropped_grid):
    img = preprocess(cropped_grid)
    img = skimage.segmentation.clear_border(img)
    img = 255 - img
    height, _ = img.shape
    square_size = height // 9
    squares = []
    for i in range(9):
        for j in range(9):
            square_img = img[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size]
            square_img = cv2.resize(square_img, (55, 55), interpolation=cv2.INTER_AREA)
            squares.append(square_img)
    return squares

# -- Machine learning model
def squares_images_to_sudoku(squares_images):
    knn = create_knn_model()
    sudoku = np.zeros((81), dtype=int)
    for i, image in enumerate(squares_images):
        sudoku[i] = predict_digit(image, knn)
    return sudoku.reshape(9, 9)

def predict_digit(img, knn):
    img_vec = img.reshape(1, -1)
    prediction = knn.predict(img_vec)[0]
    return prediction

def create_knn_model():
    df = pd.read_csv("dataset.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)
    return knn

# -- Sending the solution to the website
def solve_on_website(square_contour, solved):
    x, y, w, h = cv2.boundingRect(square_contour)
    square_size = h // 9
    for i in range(9):
        for j in range(9):
            pyautogui.click(x + j*square_size + square_size//2, y + i*square_size + square_size//2, _pause=False)
            pyautogui.press(str(solved[i, j]), _pause=False)

if __name__ == '__main__':
    main()