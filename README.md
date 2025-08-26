# CVChess: Computer Vision-Based Chess Position Recognition

## Project Overview
CVChess is an innovative computer vision system designed to recognize chess positions from images with remarkable accuracy. The project leverages deep learning and computer vision techniques to automatically detect and classify chess pieces and their positions on a chessboard, converting visual information into standard chess notation (FEN - Forsyth–Edwards Notation).

## Key Features
- Advanced chess piece recognition using deep learning
- Support for multiple viewing angles (multi-view approach)
- FEN string generation from chess position images
- Achieves 64% accuracy, which is 3 times better than previous state-of-the-art solutions
- Handles various lighting conditions and piece styles
- Integrated with standard chess notation and analysis tools

## Technical Implementation
- Built using PyTorch for deep learning
- ResNet-18 based architecture for robust feature extraction
- Bayesian optimization for hyperparameter tuning
- Custom data processing pipeline for chess position analysis
- Supports both overhead and multi-view camera angles
- Efficient batch processing of chess position images

## Project Structure
- `src/`: Source code for the chess recognition system
- `data/`: Data processing scripts and notebooks
- `reports/`: Project documentation and research findings
- Jupyter notebooks for development and testing

## Performance
- 64% accuracy in chess position recognition
- 3x improvement over previous state-of-the-art methods
- Robust performance across different:
  - Board positions
  - Lighting conditions
  - Camera angles
  - Piece designs

## Dataset
The project includes a comprehensive dataset of chess positions with:
- Multiple viewing angles
- Various game positions
- Annotated with ground truth FEN strings
- Carefully curated for training and evaluation

## Future Work
- Integration with real-time chess analysis systems
- Support for additional board and piece styles
- Mobile deployment optimization
- Enhanced multi-view recognition capabilities

## Authors
- Luthira Abeykoon

## Acknowledgments
This project was completed as part of APS360 course work, demonstrating significant advancement in computer vision applications for chess position recognition.
