#include <armadillo>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>


using namespace arma;
using namespace std;


// ********************************************************************
// ************************ ЧТЕНИЕ MNIST ******************************
// ********************************************************************


int ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_mnist_images(const std::string &filename, arma::cube &img_cube) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        int magic_number = 0, number_of_images = 0, rows = 0, cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*)&rows, sizeof(rows));
        rows = ReverseInt(rows);
        file.read((char*)&cols, sizeof(cols));
        cols = ReverseInt(cols);

        img_cube.set_size(rows, cols, number_of_images);
        for (int k = 0; k < number_of_images; ++k) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    img_cube(i, j, k) = (double)temp;
                }
            }
        }
    }
    file.close();
}


vector<unsigned char> read_mnist_labels(const string& filename) {
    ifstream file(filename, ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = __builtin_bswap32(magic_number);

        if (magic_number != 2049) {
            throw runtime_error("Поломанный файл с таргетами!");
        }

        int number_of_labels = 0;
        file.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = __builtin_bswap32(number_of_labels);

        vector<unsigned char> labels(number_of_labels);
        file.read((char*)&labels[0], number_of_labels);

        return labels;
    } else {
        throw std::runtime_error("Не открыть файл `" + filename + "`!");
    }
}


// ********************************************************************
// ************************** УТИЛИТЫ *********************************
// ********************************************************************


double get_random_double(double lb, double ub) {
    uniform_real_distribution<double> unif(lb, ub);
    default_random_engine re;

    return unif(re);
}


cube rotate180(const cube& source) {
    cube rotated(source.n_rows, source.n_cols, source.n_slices, fill::zeros);
    for (size_t i = 0; i < source.n_slices; ++i) {
        rotated.slice(i) = flipud(fliplr(source.slice(i)));
    }
    return rotated;
}


// ********************************************************************
// ************************** СВЁРТКИ *********************************
// ********************************************************************


cube add_padding(cube source, unsigned int rows_padding, unsigned int cols_padding, unsigned int slices_padding) {
    cube padded(
        source.n_rows + rows_padding * 2,
        source.n_cols + cols_padding * 2,
        source.n_slices + slices_padding * 2,
        fill::zeros
    );

    for(int row=0; row<source.n_rows; row++) {
        for(int col=0; col<source.n_cols; col++) {
            for(int k=0; k<source.n_slices; k++)
            padded(row + rows_padding, col + cols_padding, k + slices_padding) = source(row, col, k);
        }
    }

    return padded;
}


cube convolve3D_to_2D(const cube& input, const cube& kernel) {
    // сворачиваем куб кубом и получаем куб с Z-размерностью = 1
    // X, Y = input.x - kernel + 1, input.y - kernel + 1

    int input_height = input.n_rows;
    int input_width = input.n_cols;
    int input_depth = input.n_slices;

    int kernel_height = kernel.n_rows;
    int kernel_width = kernel.n_cols;
    int kernel_depth = kernel.n_slices;

    assert(input_depth == kernel_depth && "Разная Z-размерность у входа и ядра");

    int output_height = input_height - kernel_height + 1;
    int output_width = input_width - kernel_width + 1;
    cube output(output_height, output_width, 1, arma::fill::zeros);

    for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
            double sum = 0;
            for (int k = 0; k < kernel_depth; ++k) {
                for (int y = 0; y < kernel_height; ++y) {
                    for (int x = 0; x < kernel_width; ++x) {
                        sum += input(i + y, j + x, k) * kernel(y, x, k);
                    }
                }
            }
            output(i, j, 0) = sum;
        }
    }

    return output;
}


cube convolve3D_with2D(const cube& input, const cube& kernel) {
    // kernel - куб, у которого Z-размерность = 1

    int input_height = input.n_rows;
    int input_width = input.n_cols;
    int input_depth = input.n_slices;

    int kernel_height = kernel.n_rows;
    int kernel_width = kernel.n_cols;
    int kernel_depth = kernel.n_slices;

    assert(kernel_depth == 1 && "kernel должен иметь Z-размерность = 1");

    int output_height = input_height - kernel_height + 1;
    int output_width = input_width - kernel_width + 1;
    cube output(output_height, output_width, input_depth, fill::zeros);

    for (int k = 0; k < input_depth; ++k) {
        cube slice_to_convolve(input.n_rows, input.n_cols, 1, fill::zeros);
        slice_to_convolve.slice(0) = input.slice(k);
        output.slice(k) = convolve3D_to_2D(slice_to_convolve, kernel).slice(0);
    }

    return output;
}


cube convolve2D_with3D(const cube& input, const cube& kernel) {
    // input - куб, у которого Z-размерность = 1

    int input_height = input.n_rows;
    int input_width = input.n_cols;
    int input_depth = input.n_slices;

    int kernel_height = kernel.n_rows;
    int kernel_width = kernel.n_cols;
    int kernel_depth = kernel.n_slices;

    assert(input_depth == 1 && "input должен иметь Z-размерность = 1");

    int output_height = input_height - kernel_height + 1;
    int output_width = input_width - kernel_width + 1;
    cube output(output_height, output_width, kernel_depth, fill::zeros);

    for (int k = 0; k < kernel_depth; ++k) {
        cube kernel_slice_to_convolve_with(kernel.n_rows, kernel.n_cols, 1, fill::zeros);
        kernel_slice_to_convolve_with.slice(0) = kernel.slice(k);
        output.slice(k) = convolve3D_to_2D(input, kernel_slice_to_convolve_with).slice(0);
    }

    return output;
}


// ********************************************************************
// **************************** СЛОИ **********************************
// ********************************************************************


class ConvLayer {
public:
    vector<cube> kernels;
    unsigned int size;
    unsigned int depth;
    unsigned int num_kernels;
    cube last_input;
    double learning_rate;

    ConvLayer(unsigned int size, unsigned int depth, unsigned int num_kernels, double lr = 0.01) : learning_rate(lr) {
        for(int i=0; i<num_kernels; i++) {
          // ядра квадратные
          kernels.emplace_back(size, size, depth, fill::randn);
        }
        assert(size % 2 && "Требуется нечётный размер ядра");
        this->size = size; // сторона ядра нечётной длины
        this->depth = depth;
        this->num_kernels = num_kernels;
    }

    cube forward(const cube& input) {
        assert(input.n_slices == depth && "input должен иметь Z-размерность = Z-размерности ядер");

        // для обратного распространения ошибки
        last_input = input;

        int output_height = input.n_rows - size + 1;
        int output_width = input.n_cols - size + 1;
        cube convolved(output_height, output_width, num_kernels);
        for(int kernel_num=0; kernel_num<num_kernels; kernel_num++) {
            convolved.slice(kernel_num) = convolve3D_to_2D(input, kernels[kernel_num]);
        }

        return convolved;
    }

    cube backward(const cube& prev_grad) {
        assert(prev_grad.n_slices == num_kernels && "prev_grad должен иметь Z-размерность = количеству ядер");

        cube d_input = cube(last_input.n_rows, last_input.n_cols, last_input.n_slices, fill::zeros);

        for(int kernel_num=0; kernel_num<num_kernels; kernel_num++) {
            cube d_kernel = cube(size, size, depth, fill::zeros);
            cube dl(prev_grad.n_rows, prev_grad.n_cols, 1, fill::zeros);
            dl.slice(0) = prev_grad.slice(kernel_num);
            d_kernel = convolve3D_with2D(last_input, dl);
            kernels[kernel_num] -= learning_rate * d_kernel;

            // сколько нужно добавить паддинга к свернутому X со всех сторон,
            // чтобы при повторной свёртке получить результат размерности,
            // как у X
            // сторона ядра нечётной длины
            unsigned int padding_size = size - 1;
            // почему +? Сделал по чувствам, но не понятно
            d_input += convolve2D_with3D(add_padding(dl, padding_size, padding_size, 0), rotate180(kernels[kernel_num]));
        }

        return d_input;
    }
};


class MaxPullingLayer {
public:
    int size;
    vector<tuple<size_t, size_t, size_t, size_t, size_t> > mask;
    size_t in_rows;
    size_t in_cols;
    size_t in_slices;

    MaxPullingLayer(int size) {
        this->size = size;
    }

    cube forward(cube& input) {
        int in_rows = input.n_rows;
        int in_cols = input.n_cols;
        int in_slices = input.n_slices;

        this->in_rows = in_rows;
        this->in_cols = in_cols;
        this->in_slices = in_slices;

        int out_rows = in_rows / size;
        int out_cols = in_cols / size;
        int out_slices = in_slices;
        cube output(out_rows, out_cols, out_slices, fill::zeros);

        for (int i = 0; i <= in_rows - size; i += size) {
            for (int j = 0; j <= in_cols - size; j += size) {
                for(int k=0; k<in_slices; k++) {
                    mat region = input.slice(k).submat(i, j, i + size - 1, j + size - 1);
                    output(i / size, j / size, k) = region.max();
                    uword row_idx, col_idx;
                    region.max(row_idx, col_idx);
                    row_idx += i;
                    col_idx += j;
                    mask.push_back(make_tuple(row_idx, col_idx, i / size, j / size, k));
                }
            }
        }
        return output;
    }

    cube backward(const cube& prev_grad) {
        cube dx(in_rows, in_cols, in_slices, fill::zeros);
        for(auto non_null_coord : mask) {
            auto [x_row, x_col, y_row, y_col, k] = non_null_coord;
            dx(x_row, x_col, k) = prev_grad(y_row, y_col, k);
        }

        return dx;
    }
};


class FullyConnectedLayer {
public:
    mat weights;
    cube bias;
    cube input_cube;
    size_t input_rows;
    size_t input_cols;
    size_t input_slices;
    double learning_rate;


    FullyConnectedLayer(size_t input_size, size_t output_size, double lr = 0.01) : learning_rate(lr) {
        weights = mat(input_size, output_size, fill::randn) * 0.01;
        bias = cube(output_size, 1, 1, fill::zeros);
    }

    cube forward(const cube& input) {
        input_rows = input.n_rows;
        input_cols = input.n_cols;
        input_slices = input.n_slices;

        vec input_vector = vectorise(input);

        input_cube = cube(input_vector.n_elem, 1, 1);
        input_cube.slice(0).col(0) = input_vector;

        vec output_vector = weights.t() * input_vector + bias.slice(0).col(0);

        cube output_cube(output_vector.n_elem, 1, 1);
        output_cube.slice(0).col(0) = output_vector;

        return output_cube;
    }

    cube backward(const cube& grad_output) {
        vec grad_output_vec = grad_output.slice(0).col(0);

        mat grad_weights = input_cube.slice(0).col(0) * grad_output_vec.t();
        vec grad_bias = grad_output_vec;

        weights -= learning_rate * grad_weights;
        bias.slice(0).col(0) -= learning_rate * grad_bias;

        vec grad_input = weights * grad_output_vec;

        cube grad_input_cube = cube(grad_input.n_elem, 1, 1);
        grad_input_cube.slice(0).col(0) = grad_input;

        grad_input_cube.reshape(input_rows, input_cols, input_slices);

        return grad_input_cube;
    }
};


class SoftmaxLayer {
public:
    cube forward(const cube& input) {
        vec x = input.slice(0).col(0);

        double maxElem = x.max();

        vec exps = exp(x - maxElem);

        vec output = exps / accu(exps);

        cube result(10, 1, 1);
        result.slice(0).col(0) = output;
        return result;
    }

    cube backward(const cube& grad_output, const cube& forward_output) {
        vec y = forward_output.slice(0).col(0);
        vec gy = grad_output.slice(0).col(0);

        mat J = diagmat(y) - y * y.t();

        vec grad_input = J * gy;

        cube result(grad_input.n_elem, 1, 1);
        result.slice(0).col(0) = grad_input;
        return result;
    }

    size_t predict(const cube& softmax_output) {
        vec probabilities = softmax_output.slice(0).col(0);
        uword predicted_class;
        probabilities.max(predicted_class);
        return predicted_class;
    }
};


// ********************************************************************
// *********************** Функция потерь *****************************
// ********************************************************************


cube compute_loss_and_gradient(const cube& softmax_output, int true_label) {
    vec true_vec = zeros<vec>(10);
    true_vec(true_label) = 1;

    vec predicted_probabilities = softmax_output.slice(0).col(0);
    double loss = -dot(true_vec, log(predicted_probabilities));

    vec gradient = predicted_probabilities - true_vec;

    cout << endl << "*************************\nКросс-энтропия: " << loss << "\n*************************" << endl;

    cube grad_cube(10, 1, 1);
    grad_cube.slice(0).col(0) = gradient;

    return grad_cube;
}


int main() {
    cube images;

    read_mnist_images("train-images-idx3-ubyte", images);

    string filename = "train-labels-idx1-ubyte";
    vector<unsigned char> labels;
    try {
        labels = read_mnist_labels(filename);
        cout << "Прочитано " << labels.size() << " строк." << endl;
        cout << "Первый класс: " << static_cast<int>(labels[0]) << endl;
    } catch (const runtime_error& e) {
        cerr << "Ошибка: " << e.what() << endl;
        return EXIT_FAILURE;
    }

    /*
    LeNet-5
      - Layer 1: Свёртка из 6 ядер 5*5.
      - Layer 2: Субдискретизация 2*2.
      - Layer 3: Свёртка из 6 ядер 5*5.
      - Layer 4: Субдискретизация 2*2.
      - Layer 5: Полносвязный слой.
      - Layer 6: Полносвязный слой.
      - Output Layer: A softmax layer with 10 outputs.
    */

    ConvLayer conv1(5, 1, 6);  // 28*28*1 ==> 24*24*6
    MaxPullingLayer pull1(2);  // 24*24*6 ==> 12*12*6
    ConvLayer conv2(5, 6, 16);  // 12*12*6 ==> 8*8*16
    MaxPullingLayer pull2(2);  // 8*8*16 ==> 4*4*16
    FullyConnectedLayer fc1(4*4*16, 10);  // 4*4*16 ==> 10
    SoftmaxLayer sf;  // 10 ==> 10

    for(int i=0; i<10; i++) {  // images.n_slices; i++) {
        cube first_image(images.n_rows, images.n_cols, 1);
        first_image.slice(0) = images.slice(i);
        cube conv1_result = conv1.forward(first_image);
        cube pull1_result = pull1.forward(conv1_result);
        cube conv2_result = conv2.forward(pull1_result);
        cube pull2_result = pull2.forward(conv2_result);
        cube fc1_result = fc1.forward(pull2_result);
        cube sf_result = sf.forward(fc1_result);
        sf_result.print("Результат:");

        size_t predicted_class = sf.predict(sf_result);

        int real_label = static_cast<int>(labels[i]);
        cout << "Предсказанное число: " << predicted_class << ". Реально число: " << real_label << endl;

        cube grad_from_loss = compute_loss_and_gradient(sf_result, real_label);
        // grad_from_loss.print("grad_from_loss");
        cube sf_grad = sf.backward(grad_from_loss, sf_result);
        // sf_grad.print("sf_grad");
        cube fc1_grad = fc1.backward(sf_grad);
        cube pull2_grad = pull2.backward(fc1_grad);
        cube conv2_grad = conv2.backward(pull2_grad);
        cube pull1_grad = pull1.backward(conv2_grad);
        conv1.backward(pull1_grad);
        cout << "\n----------------------------------------------------------\n\n\n";
    }

    return 0;
}
