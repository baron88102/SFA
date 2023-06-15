#include <Rcpp.h> // Headers Rcpp
#include <vector> // vector management
#include <numeric> // math functions
#include <algorithm> // Algorithms
#include <cmath> // math functions
#include <tuple> // Tuple data structure. (value1, value2,...)
#include <complex> // Handling complex numbers
#include <valarray> // value array

using namespace Rcpp;
using std::string;
using std::cout;
using std::endl;
using std::vector;
using std::accumulate;
using std::cos;
using std::sin;
typedef std::complex<double> Complex;       
typedef std::valarray<Complex> CArray;      





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// Fast Fourier Transform ///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//' Converts a vector of numbers to letters
//'
//' This function is used to convert the result of searching the buckets to Fourier coefficients.
//'
//' @param word Contains the bucket indices to create SFA.
//' @return SFA word.
void fft(CArray &x)
{
  // Extract the real part of the whole numbers.
  vector<float> serie(x.size());
  int end_for = static_cast<int>(x.size());
  for (int i=0 ; i < end_for; ++i) {
    serie[i] = x[i].real();
  }
  
  // Obtain the namespace of the stats package. The namespace is assigned to "stats_env".
  Environment stats_env = Environment::namespace_env("stats");
  
  // From the above defined "stats_env", we get function "fft" and  assign it to "stats_fft_in_cpp".
  Function stats_fft_in_cpp = stats_env["fft"];
  
  // Calculate the Fourier transform
  SEXP res = stats_fft_in_cpp(serie);
  
  // Store the FFT response in the vector passed by reference
  ComplexVector res2(res);
  end_for = static_cast<int>(res2.size());
  for (int i = 0 ; i < end_for ; ++i) {
    Rcomplex t = res2[i];
    Complex c(t.r, t.i);
    x[i] = c;
  }
}




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// TimeSeries Class /////////////////////////////////////////////////////////////
// This class stores the values and characteristics of the time series passed to the sfa_cpp function.
// This class is particularly useful when developing WEASEL, BOSS, TEASER classifier algorithms
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//' @name TimeSeries
//' @title Encapsulates a time series
//' @description This class represents a time series (numeric vector) and stores some of its characteristics.
//' @field TimeSeries Parametric constructor \itemize{
//' \item data: Numeric vector with the time series.
//' \item label: Indicates a label for the time series. Used with the classification algorithms for your training.
//' \item NORM_CHECK: Indicates if the normalization of the series should be verified.
//' \item APPLY_Z_NORM: Indicates whether or not the series should be normalized
//' }
//' @field ~TimeSeries Destructor
//' @field show_data Display the time series data
//' @field show_attributes Shows the characteristics of the time series
//' @field NORM Normalize the time series \itemize{
//' \item norm_mean: Indicates whether or not the string should be normalized
//' }
//' @field calculate_std Calculate the standard deviation of the time series.
//' @field NORM_WORK Do the work of normalizing the string. \itemize{
//' \item norm_mean: Indicates whether or not the string should be normalized
//' }
//' @field getSubsequence Get a sebsequence from the time series. \itemize{
//' \item offset: Indicates the number of values to skip at the beginning of the subsequence
//' \item window_size: number of values in the substring.
//' }
//' @field size Returns the size of the time series.
class TimeSeries {
public:
  TimeSeries(vector<float> data, string label, bool NORM_CHECK = true, bool APPLY_Z_NORM = true);
  ~TimeSeries();
  void show_data() const;
  void show_attributes() const;
  void NORM(bool norm_mean = true);
  void calculate_std(); 
  void NORM_WORK(bool norm_mean);
  TimeSeries getSubsequence(int offset, int window_size); 
  int size() const;
  
private:
  bool NORM_CHECK;
  bool APPLY_Z_NORM;
  bool normed;
  float mean;
  float std;
public:
  std::string label;
  vector<float> *data;
};


TimeSeries::TimeSeries(vector<float> data, string label, bool NORM_CHECK, bool APPLY_Z_NORM) {
  this->data = new vector<float>(data.size());
  std::copy(data.begin(), data.end(), this->data->begin()) ;
  this->label = label;
  this->normed = false;
  this->mean = 0;
  this->std = 1;
  this->APPLY_Z_NORM = APPLY_Z_NORM;
  this->NORM_CHECK = NORM_CHECK;
}


TimeSeries::~TimeSeries() {
  if (this->data != 0) {
    delete this->data;
    this->data = 0;
  }
}

  
int TimeSeries::size() const {
  return (this->data->size());
}


void TimeSeries::show_data() const {
  int end_for = static_cast<int>(this->data->size());
  for (int i = 0; i < end_for; ++i) {
    cout << this->data->at(i) << "\t";
  }
  cout << endl;
}


void TimeSeries::show_attributes() const {
  cout << "series attributes: " << endl;
  cout << "label: "         << "\t" << "\t" << this->label << endl;
  cout << "normed: "        << "\t" << this->normed << endl;
  cout << "mean: "          << "\t" << "\t" << this->mean << endl;
  cout << "std: "           << "\t" << "\t" << this->std << endl;
  cout << "APPLY_Z_NORM: "  << "\t" << this->APPLY_Z_NORM << endl;
  cout << "NORM_CHECK: "    << "\t" << this->NORM_CHECK << endl;
}


void TimeSeries::NORM(bool norm_mean) {
  // Calculate the mean of the series
  this->mean = (std::accumulate(this->data->begin(), this->data->end(), 0.0)) / this->data->size();

  // Calculate the standard deviation
  this->calculate_std();

  // If the time series passed to the constructor is not normalized then we must normalize.
  if (!this->normed) {
    this->NORM_WORK(norm_mean);
  }
}


void TimeSeries::calculate_std() {
  float var = 0.0;
  int end_for = static_cast<int>(this->data->size());
  for (int i = 0; i < end_for; ++i) {
    var += this->data->at(i) * this->data->at(i);
  }

  // Calculate the standard deviation
  float norm = 1.0 / this->data->size();

  float buf = (norm * var) - (this->mean * this->mean);
  buf = std::abs(buf);

  if (buf != 0.0) {
    this->std = sqrt(buf); 
  } else {
    this->std = 0;
  }
}


void TimeSeries::NORM_WORK(bool norm_mean) {
  if (this->APPLY_Z_NORM & (!this->normed)) {
    // Factor used in the normalization that depends on the value of std
    float ISTD = 0;
    if (this->std == 0) {
      ISTD = 1;
    } else {
      ISTD = 1.0 / this->std;
    }

    if (norm_mean) {
      // Z normalization
      int end_for = static_cast<int>(this->data->size());
      for (int i = 0; i < end_for; ++i) {
        this->data->at(i) = (this->data->at(i) - this->mean) * ISTD;
      }
      this->mean = 0.0;
    } else {
      if (ISTD != 1.0) {
        int end_for = static_cast<int>(this->data->size());
        for (int i = 0; i < end_for; ++i) {
          this->data->at(i) = this->data->at(i) * ISTD;
        } 
      }
    }
    this->normed = true;
  }
}  


TimeSeries TimeSeries::getSubsequence(int offset, int window_size) {
  // sample subsequence
  vector<float> subvector = {this->data->begin() + offset, this->data->begin() + (offset + window_size)};
  
  // Create a new time series with the subsequence
  TimeSeries subseqences_data = TimeSeries(subvector,
                                           this->label, 
                                           this->NORM_CHECK, 
                                           this->APPLY_Z_NORM);
  // Check if the series is normalized.
  subseqences_data.NORM();
  
  return (subseqences_data);
}




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// Class MFT        /////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//' @name MFT
//' @title Encapsulates a time series
//' @description The Momentary Fourier Transform is an alternative algorithm of
//' the Discrete Fourier Transform for overlapping windows. it has
//' a constant computational complexity for in the window queryLength n as
//' opposed to Onlogn for the Fast Fourier Transform algorithm.
//' @field MFT Parametric constructor \itemize{
//' \item windowSize: Size of the window.
//' \item normMean: Indicates whether to set startOffset.
//' \item lowerBounding: Used to define how normalization is calculated.
//' \item MUSE_Bool: Indicates whether to use MUSE
//' }
//' @field ~MFT Destructor
//' @field transform: Apply the fourier transformation on the time series \itemize{
//' \item series: Time series
//' \item wordlength: Length of the SFA word.
//' }
//' @field show: Show the values of the parameters of the class.
class MFT {
public:
  MFT(int windowSize, bool normMean, bool lowerBounding, bool MUSE_Bool = false);
  ~MFT();
  vector<float> transform(vector<float> &series, int wordlength);
  void show();
private:
  int windowSize;
  bool MUSE;
  int startOffset;
  float norm;
};


MFT::MFT(int window_size, bool normMean, bool lowerBounding, bool MUSE_Bool) {
  this->windowSize = window_size;
  this->MUSE = MUSE_Bool;
  if (normMean) {
    this->startOffset = 2;
  } else {
    this->startOffset = 0;
  }
  if (lowerBounding) {
    this->norm = 1.0 / sqrt(window_size);
  } else {
    this->norm = 1.0;
  }
}


MFT::~MFT() {
  
}  


vector<float> MFT::transform(vector<float> &series, int wordlength) {
  // Convert the string to complex numbers
  CArray data((int)series.size());
  int end_for = static_cast<int>(series.size());
  for (int i = 0; i < end_for; ++i) {
    data[i] = Complex(series[i]);
  }

  // Calculate the Fourier Transform
  fft(data);

  // Convert the string to complex numbers
  vector<float> data_new;
  int windowSize = series.size();
  end_for = int(ceil(windowSize / 2));

  for (int i = 0; i < end_for; ++i) {
    data_new.push_back(data[i].real());
    data_new.push_back(data[i].imag());
  }

  // DC-coefficient imaginary part
  data_new[1] = 0.0;

  // Make it even length for uneven windowSize
  vector<float> data_new_slice(this->windowSize);
  std::copy(data_new.begin(), data_new.begin() + this->windowSize, data_new_slice.begin());
  int length = std::min(windowSize - this->startOffset, wordlength);
  vector<float> copy(length);
  std::copy(data_new_slice.begin() + this->startOffset,
            data_new_slice.begin() + length + this->startOffset,
            copy.begin());
  while ((int)copy.size() != wordlength) {
    copy.push_back(0.0);
  }

  // norming
  int sign = 1;
  end_for = static_cast<int>(copy.size());
  for(int i = 0; i < end_for; ++i) {
    copy[i] *= this->norm * sign;
    sign *= -1;
  }
  
  return (copy);
}


void MFT::show() {
  cout << " MFT values" << endl;
  cout << ": " << windowSize << endl;
  cout << ": " << MUSE << endl;
  cout << ": " << startOffset << endl;
  cout << ": " << norm << endl;
}




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// TimeSeries SFA   /////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//' @name SFA
//' @title Symbolic Fourier Approximation
//' @description Symbolic Fourier Approximation.
//' @field SFA: Parametric constructor \itemize{
//' \item histogram_type:
//' \item LB: Indicates whether to use lower bounding in MFT.
//' \item mft_use_max_or_min: Indicates whether to use maximum or minimum in MFT.
//' }
//' @field ~SFA Destructor
//' @field initialize: Normalize the time series \itemize{
//' \item wordLength: Sets the value of the word length.
//' \item symbols: Indicates the size of the alphabet to use.
//' \item norm_mean: Indicates the normalization.
//' }
//' @field print_bins: Print the bins for the transformation.
//' @field fitTransform Calls helper functions for the transformation. Returns the SFA words. \itemize{
//' \item samples: Object of the time series class.
//' \item word_length: Length of the word.
//' \item symbols: Size of the alphabet.
//' \item norm_mean: Indicates normalization.
//' }
//' @field fitTransformDouble: Performs the transformation of the time series to the histogram positions of the Fourier coefficients. \itemize{
//' \item samples: Object of the time series class.
//' \item word_length: Length of the word.
//' \item symbols: Size of the alphabet.
//' \item norm_mean: Indicates normalization.
//' }
//' @field fillOrderline: Create the interval divisions for the Fourier coefficients. \itemize{
//' \item samples: Object of the time series class.
//' \item word_length: Length of the word.
//' }
//' @field divideEquiWidthHistogram: Divide the histogram into equally sized sections.
//' @field transform: Converts the position vector of Fourier coefficients to letters of the alphabet. \itemize{
//' \item series: Representation in the form of a vector of positions of the time series.
//' \item approximate: Matrix with the buckets for the approximations.
//' }
//' @field transform2: Create the SFA words from their approximations. \itemize{
//' \item series: Representation in the form of a vector of positions of the time series.
//' \item one_approx: Matrix with the buckets for the approximations.
//' \item str_return: Indicates whether the result is returned as a string or not.
//' }
//' @field quantization: From the vector of approximations it looks for its corresponding letter of the alphabet. \itemize{
//' \item one_approx: Vector with the approximations of the time series.
//' }
class SFA {
public:
  SFA(string histogram_type, bool LB = true, bool mft_use_max_or_min = false);
  ~SFA();
  void initialize(int wordLength, int symbols, bool norm_mean);  
  void print_bins() const;
  vector<std::string> fitTransform(TimeSeries &samples, int word_length, int symbols, bool norm_mean);
  vector<vector<float> > fitTransformDouble(TimeSeries &samples, int word_length, int symbols, bool norm_mean);
  vector<vector<float> > fillOrderline(TimeSeries &samples, int word_length);
  void divideEquiWidthHistogram(); 
  vector<std::string> transform(vector<float> &series, vector<vector<float> > approximate); 
  std::string transform2(vector<float> &series, vector<vector<float> > one_approx, bool str_return = true); 
  vector<int> quantization(vector<float> &one_approx); 
  std::string sfaToWord(const std::vector<int> &word);  
private:
  bool initialized;
  string HistogramType;
  bool lowerBounding;
  bool mftUseMaxOrMin;
  int wordLength;
  int maxWordLength;
  int symbols;
  bool normMean;
  int alphabetSize;
public:
  MFT *transformation;
  vector<vector<float>> *bins;
  vector<vector<std::tuple<float, float> > >  orderLine;
};


std::string SFA::sfaToWord(const std::vector<int> &word) {
  // SFA word
  std::string word_string = "";
  
  // basic alphabet
  std::string alphabet = "abcdefghijklmnopqrstuv";
  
  // Convert each position of the vector into a letter.
  int end_for = static_cast<int>(word.size());
  for (int i = 0; i < end_for; ++i) {
    word_string += alphabet.substr(word[i], 1);
  }
  
  return (word_string);
}



SFA::SFA(string histogram_type, bool LB, bool mft_use_max_or_min) {
  this->initialized = false;
  this->HistogramType = histogram_type;
  this->lowerBounding = LB;
  this->mftUseMaxOrMin = mft_use_max_or_min;
  this->transformation = 0;
  this->bins = 0;
}


SFA::~SFA() {
  if (this->transformation != 0) {
    delete this->transformation;
    this->transformation = 0;
  }

  if (this->bins != 0) {
    delete this->bins;
    this->bins = 0;
  }
}


void SFA::initialize(int word_length, int symbols, bool norm_mean) {
  this->initialized = true;
  this->wordLength = word_length;
  this->maxWordLength = word_length;
  this->symbols = symbols;
  this->normMean = norm_mean;
  this->alphabetSize = symbols;
  this->transformation = 0;
  
  // Create the buckets for the coefficients
  bins = new vector<vector<float>>(this->alphabetSize);
  vector<float> v(word_length, -INFINITY);
  bins->at(0) = v;

  for (int i = 1; i < this->alphabetSize; ++i) {
    vector<float> v(word_length, INFINITY);
    bins->at(i) = v;
  }
}


void SFA::print_bins() const {
  // Show buckets
  for (int j = 0; j < this->alphabetSize; ++j) {
    int end_for = static_cast<int>(this->bins[0][j].size());
    for (int k = 0; k < end_for; ++k) {
      cout << this->bins->at(j)[k] << " ";
    }
    cout << endl;
  }
}


vector<std::string> SFA::fitTransform(TimeSeries &samples, int word_length, int symbols, bool norm_mean) {
  // Create the time series approximation
  vector<vector<float> > approximate = this->fitTransformDouble(samples, word_length, symbols, norm_mean);

  // Create the word SFA
  vector<std::string> words = this->transform(*samples.data, approximate);

  return (words);
}


vector<vector<float> > SFA::fitTransformDouble(TimeSeries &samples, int word_length, int symbols, bool norm_mean) { 
  // Check if the class has been initialized and that the object for the transformation has been created
  if (this->initialized == false) {
    this->initialize(word_length, symbols, norm_mean);
    if (this->transformation == 0) {
      this->transformation = new MFT(samples.size(), norm_mean, this->lowerBounding);
    }
  }

  // The buckets (bins) are created and the corresponding histogram is called for their division
  vector<vector<float> > transformedSamples = this->fillOrderline(samples, word_length);

  if (this->HistogramType == "EQUI_FREQUENCY") {
    this->divideEquiWidthHistogram();
  }

  this->orderLine.clear();

  return transformedSamples;
}


// Time Series & samples: This parameter should be a vector of TimeSeries if you want to develop the classification algorithms
vector<vector<float> > SFA::fillOrderline(TimeSeries &samples, int word_length) { 
  // Create the tuples containing the fourier coefficient and the string label.
  // The series label is used when developing the classification algorithms
  for (int i = 0; i < word_length; ++i) {
    std::tuple <float, float> t;
    t = std::make_tuple(0.0, 0.0);
    vector<std::tuple<float, float> > vect_tuple;
    vect_tuple.push_back(t);
    this->orderLine.push_back(vect_tuple);
  }
  
  // Stores the transformed time series
  vector<vector<float> > transformedSamples;

  // Apply the Fourier transformation to the series
  vector<float> transformedSamples_small = this->transformation->transform(*samples.data, wordLength);

  // The transformed series is converted into tuples (pairs)
  transformedSamples.push_back(transformedSamples_small);

  int end_for = static_cast<int>(transformedSamples_small.size());
  for (int j = 0; j < end_for; ++j) {
    float value = std::round(transformedSamples_small[j] * 100.0) / 100.0 ;
    std::tuple <float, float> obj;
    obj = std::make_tuple(value, std::stof(samples.label));
    this->orderLine[j][0] = obj;
  }

  // The values of the limits of the buckets are updated based on the transformation
  end_for = static_cast<int>(this->orderLine.size());
  for (int i = 0; i < end_for; ++i) {
    vector<std::tuple<float, float> > del_list = this->orderLine[i];
    vector<std::tuple<float, float> > new_list;
    while ((int)del_list.size() != 0) {
      float current_min_value = INFINITY;
      int current_min_location = -1;
      float label = -INFINITY;

      int end_for2 = static_cast<int>(del_list.size());
      for(int j = 0; j < end_for2; ++ j) {
        if ((std::get<0>(del_list[j]) < current_min_value) |
            ((std::get<0>(del_list[j]) == current_min_value) & (std::get<1>(del_list[j]) < label))) {
          current_min_value = std::get<0>(del_list[j]);
          label = std::get<1>(del_list[j]);
          current_min_location = j;
        }
      }
      new_list.push_back(del_list[current_min_location]);
      del_list.erase(del_list.begin() + current_min_location);
    }
    this->orderLine[i] = new_list;
  }

  return (transformedSamples);
}


void SFA::divideEquiWidthHistogram() {
  // All pairs of buckets are traversed
  int end_for = static_cast<int>(this->orderLine.size());
  for (int i = 0; i < end_for; ++i) {
    vector<std::tuple<float, float> > element = this->orderLine[i];

    // Make all gaps between buckets equal to a fixed size
    if (element.size() > 0) {
      float first = std::get<0>(element.front());
      float last = std::get<0>(element.back());
      float intervalWidth = (last - first) / this->alphabetSize;
      int end_for2 = this->alphabetSize - 1;
      for (int j = 0; j < end_for2; ++j) {
        this->bins->at(j+1)[i] = intervalWidth * (j + 1) + first;
      }
    }
  }
}


vector<std::string> SFA::transform(vector<float> &series, vector<vector<float> > approximate) {
  vector<std::string> words;
  
  words.push_back(this->transform2(series, approximate)); 
  
  return (words);
}


std::string SFA::transform2(vector<float> &series, vector<vector<float> > one_approx, bool str_return) {
  vector<float> one_approx_2;
  std::string word;
  
  // Check if the bucket approximation exists or not.
  if (one_approx.size() == 0) {
    one_approx_2 = this->transformation->transform(series, this->maxWordLength);
    
    if (str_return) {
      word = this->sfaToWord(this->quantization(one_approx_2));
    } else {
      //return (this->quantization(one_approx));  
    }
  } else {
    if (str_return) {
      word = this->sfaToWord(this->quantization(one_approx[0]));
    } else {
      //return (this->quantization(one_approx));  
    }
  }
  
  return (word);
}  


vector<int> SFA::quantization(vector<float> &one_approx) {
  vector<int> word(one_approx.size(), 0);

  // Loop through all buckets for a series and compute the corresponding bucket position which will then be transformed into a letter.
  int end_for = static_cast<int>(one_approx.size());
  for (int i = 0; i < end_for; ++i) {
    int c = 0;
    int end_for2 = static_cast<int>(this->bins->size());
    for (int j = 0 ; j < end_for2; ++j) {
      if (one_approx[i] < this->bins->at(j).at(i)) {
        break;
      } else {
        c += 1;
      }
    }
    word[i] = c - 1;
  }  
  
  return (word);
}  




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// Support functions for processing the data ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Function to validate the input parameters to the sfa function
bool validate_parameters(const std::string histogram_type, int word_length, int symbols, int norm_mean, int MAX_WINDOW_LENGTH, int sliding_window, int serie_size) {
  string error_message = "Errors in parameters: ";
  bool valid_parameters = true;
  
  // histogram_type
  bool finded = false;
  std::vector<std::string> histogram_types = {"INFORMATION_GAIN", "EQUI_FREQUENCY"};
  finded = std::find(histogram_types.begin(), histogram_types.end(), histogram_type) != histogram_types.end();
  if (!finded) {
    error_message = error_message + "histogram_type must be 'EQUI_FREQUENCY' ";
    valid_parameters = false;
  }
  
  // word_length
  if ((word_length < 1) | (word_length > 20)) {
    error_message = error_message + "word_length must be greather than 0 and less equal 20";
    valid_parameters = false;
  }

  // symbols
  if ((symbols < 3) | (symbols > 22)) {
    error_message = error_message + "symbols must be greather than 2 and less equal 22";
    valid_parameters = false;
  }
  
  // sliding_window
  if ((sliding_window > serie_size)) {
    error_message = error_message + "sliding_windows must be less equal serie.size";
    valid_parameters = false;
  }

  if ((sliding_window < word_length)) {
    error_message = error_message + "sliding_windows must be greather equal word.length";
    valid_parameters = false;
  }
  
  if (valid_parameters == false) {
    cout << error_message;
  }
  
  return (valid_parameters);
} 


//' @name sfa_function
//' @title Constructs a new Double object
//' @param serie time serie to convert to word.
//' @param sw time se convert to word.
//' @param wl tim serie to convert to word.
//' @param as time see to convert to word.
//' @param norm_mean ime serie to convert to word.
//' @param LowerBounding time se convert to word.
//' @param mft_use_max_or_min tito convert to word.
//' @param norm_check time serie to c to word.
//' @param apply_znorm time serie to conveword.
//' @param normalize time serie to convert to d.
//' @return SFA words
// [[Rcpp::export("fn_sfa")]]
CharacterVector fn_sfa(NumericVector data, int sliding_window = 120, int wordlength = 6, int alphabet_size = 5, bool norm_mean = true, bool apply_znorm = true) {
  bool LowerBounding = true;
  bool mft_use_max_or_min = false;
  bool norm_check = true;
  
  // int sliding_window_aux = Rcpp::as<int>(sliding_window); 
  vector<float> serie_aux = Rcpp::as< std::vector<float> >(data);
  
  // ------------------------------ Read and validate the parameters ------------- -----------------
  std::string histogram_type = "EQUI_FREQUENCY";
  int symbols = alphabet_size; 
  int MAX_WINDOW_LENGTH = wordlength;
  string label = "1";
  
  bool validated_parameters = validate_parameters(histogram_type, wordlength, symbols, norm_mean, MAX_WINDOW_LENGTH, sliding_window, (int)serie_aux.size());
  if (!validated_parameters) {
    return (CharacterVector());
  }
  
  
  // ------------------------------ Read training and test strings ----------- -------------------
  // ------ When working with a single time series, this is the same for training and validation
  
  // Execute the transformation
  int total_runs = (int)serie_aux.size() - sliding_window + 1; // Total Sliding Window Runs
  vector<std::string> words_sfa(total_runs); // Contains the resulting SFA words

  for (int i = 0; i < total_runs; i++) {  // 1
    vector<float> serie_sample(sliding_window);
    
    // Create a TimeSeries object with the part of the series that corresponds to the window
    for (int k = i; k <= (i + sliding_window - 1); ++k) {
      serie_sample[k - i] = serie_aux[k];
    }

    // serie_train_cpp = new TimeSeries(serie_sample, label, norm_check, apply_znorm);
    TimeSeries serie_train_cpp(serie_sample, label, norm_check, apply_znorm);
    if (apply_znorm) {
      serie_train_cpp.NORM(norm_mean);
    }

    // Perform the transformation for the sliding window
    SFA sfa_object(histogram_type, LowerBounding, mft_use_max_or_min);

    words_sfa.at(i) = sfa_object.fitTransform(serie_train_cpp, wordlength, symbols, norm_mean)[0];
  }
  
  return (Rcpp::wrap(words_sfa));
}
