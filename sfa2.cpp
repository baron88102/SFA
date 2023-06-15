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
void FFT(CArray &x)
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
//' \item norm_check: Indicates if the normalization of the series should be verified.
//' \item apply_z_norm: Indicates whether or not the series should be normalized
//' }
//' @field ~TimeSeries Destructor
//' @field ShowData Display the time series data
//' @field ShowAttributes Shows the characteristics of the time series
//' @field Norm Normalize the time series \itemize{
//' \item norm_mean: Indicates whether or not the string should be normalized
//' }
//' @field CalculateStd Calculate the standard deviation of the time series.
//' @field NormWork Do the work of normalizing the string. \itemize{
//' \item norm_mean: Indicates whether or not the string should be normalized
//' }
//' @field GetSubSequence Get a sebsequence from the time series. \itemize{
//' \item offset: Indicates the number of values to skip at the beginning of the subsequence
//' \item window_size: number of values in the substring.
//' }
//' @field Size Returns the size of the time series.
class TimeSeries {
public:
  TimeSeries(vector<float> data, string label, bool norm_check = true, bool apply_z_norm = true);
  ~TimeSeries();
  void ShowData() const;
  void ShowAttributes() const;
  void Norm(bool norm_mean = true);
  void CalculateStd(); 
  void NormWork(bool norm_mean);
  TimeSeries GetSubSequence(int offset, int window_size); 
  int Size() const;
  
private:
  bool norm_check;
  bool apply_z_norm;
  bool normed;
  float mean;
  float std;
public:
  std::string label;
  vector<float> *data;
};


TimeSeries::TimeSeries(vector<float> data, string label, bool norm_check, bool apply_z_norm) {
  this->data = new vector<float>(data.size());
  std::copy(data.begin(), data.end(), this->data->begin()) ;
  this->label = label;
  this->normed = false;
  this->mean = 0;
  this->std = 1;
  this->apply_z_norm = apply_z_norm;
  this->norm_check = norm_check;
}


TimeSeries::~TimeSeries() {
  if (this->data != 0) {
    delete this->data;
    this->data = 0;
  }
}

  
int TimeSeries::Size() const {
  return (this->data->size());
}


void TimeSeries::ShowData() const {
  int end_for = static_cast<int>(this->data->size());
  for (int i = 0; i < end_for; ++i) {
    cout << this->data->at(i) << "\t";
  }
  cout << endl;
}


void TimeSeries::ShowAttributes() const {
  cout << "series attributes: " << endl;
  cout << "label: "         << "\t" << "\t" << this->label << endl;
  cout << "normed: "        << "\t" << this->normed << endl;
  cout << "mean: "          << "\t" << "\t" << this->mean << endl;
  cout << "std: "           << "\t" << "\t" << this->std << endl;
  cout << "apply_z_norm: "  << "\t" << this->apply_z_norm << endl;
  cout << "norm_check: "    << "\t" << this->norm_check << endl;
}


void TimeSeries::Norm(bool norm_mean) {
  // Calculate the mean of the series
  this->mean = (std::accumulate(this->data->begin(), this->data->end(), 0.0)) / this->data->size();

  // Calculate the standard deviation
  this->CalculateStd();

  // If the time series passed to the constructor is not normalized then we must normalize.
  if (!this->normed) {
    this->NormWork(norm_mean);
  }
}


void TimeSeries::CalculateStd() {
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


void TimeSeries::NormWork(bool norm_mean) {
  if (this->apply_z_norm & (!this->normed)) {
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


TimeSeries TimeSeries::GetSubSequence(int offset, int window_size) {
  // sample subsequence
  vector<float> subvector = {this->data->begin() + offset, this->data->begin() + (offset + window_size)};
  
  // Create a new time series with the subsequence
  TimeSeries subseqences_data = TimeSeries(subvector,
                                           this->label, 
                                           this->norm_check, 
                                           this->apply_z_norm);
  // Check if the series is normalized.
  subseqences_data.Norm();
  
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
//' \item window_size: Size of the window.
//' \item norm_mean: Indicates whether to set start_offset.
//' \item lower_bounding: Used to define how normalization is calculated.
//' \item muse_bool: Indicates whether to use muse
//' }
//' @field ~MFT Destructor
//' @field Transform: Apply the fourier transformation on the time series \itemize{
//' \item series: Time series
//' \item word_length: Length of the SFA word.
//' }
//' @field Show: Show the values of the parameters of the class.
class MFT {
public:
  MFT(int window_size, bool arg_norm_mean, bool lower_bounding, bool muse_bool = false);
  ~MFT();
  vector<float> Transform(vector<float> &series, int word_length);
  void Show();
private:
  int window_size;
  bool muse;
  int start_offset;
  float norm;
};


MFT::MFT(int window_size, bool norm_mean, bool lower_bounding, bool muse_bool) {
  this->window_size = window_size;
  this->muse = muse_bool;
  if (norm_mean) {
    this->start_offset = 2;
  } else {
    this->start_offset = 0;
  }
  if (lower_bounding) {
    this->norm = 1.0 / sqrt(window_size);
  } else {
    this->norm = 1.0;
  }
}


MFT::~MFT() {
  
}  


vector<float> MFT::Transform(vector<float> &series, int word_length) {
  // Convert the string to complex numbers
  CArray data((int)series.size());
  int end_for = static_cast<int>(series.size());
  for (int i = 0; i < end_for; ++i) {
    data[i] = Complex(series[i]);
  }

  // Calculate the Fourier Transform
  FFT(data);

  // Convert the string to complex numbers
  vector<float> data_new;
  int window_size = series.size();
  end_for = int(ceil(window_size / 2));

  for (int i = 0; i < end_for; ++i) {
    data_new.push_back(data[i].real());
    data_new.push_back(data[i].imag());
  }

  // DC-coefficient imaginary part
  data_new[1] = 0.0;

  // Make it even length for uneven window_size
  vector<float> data_new_slice(this->window_size);
  std::copy(data_new.begin(), data_new.begin() + this->window_size, data_new_slice.begin());
  int length = std::min(window_size - this->start_offset, word_length);
  vector<float> copy(length);
  std::copy(data_new_slice.begin() + this->start_offset,
            data_new_slice.begin() + length + this->start_offset,
            copy.begin());
  while ((int)copy.size() != word_length) {
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


void MFT::Show() {
  cout << " MFT values" << endl;
  cout << ": " << window_size << endl;
  cout << ": " << muse << endl;
  cout << ": " << start_offset << endl;
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
//' @field Initialize: Normalize the time series \itemize{
//' \item word_length: Sets the value of the word length.
//' \item symbols: Indicates the size of the alphabet to use.
//' \item norm_mean: Indicates the normalization.
//' }
//' @field PrintBins: Print the bins for the transformation.
//' @field FitTransform Calls helper functions for the transformation. Returns the SFA words. \itemize{
//' \item samples: Object of the time series class.
//' \item word_length: Length of the word.
//' \item symbols: Size of the alphabet.
//' \item norm_mean: Indicates normalization.
//' }
//' @field FitTransformDouble: Performs the transformation of the time series to the histogram positions of the Fourier coefficients. \itemize{
//' \item samples: Object of the time series class.
//' \item word_length: Length of the word.
//' \item symbols: Size of the alphabet.
//' \item norm_mean: Indicates normalization.
//' }
//' @field FillOrderline: Create the interval divisions for the Fourier coefficients. \itemize{
//' \item samples: Object of the time series class.
//' \item word_length: Length of the word.
//' }
//' @field DivideEquiWidthHistogram: Divide the histogram into equally sized sections.
//' @field Transform: Converts the position vector of Fourier coefficients to letters of the alphabet. \itemize{
//' \item series: Representation in the form of a vector of positions of the time series.
//' \item approximate: Matrix with the buckets for the approximations.
//' }
//' @field Transform2: Create the SFA words from their approximations. \itemize{
//' \item series: Representation in the form of a vector of positions of the time series.
//' \item one_approx: Matrix with the buckets for the approximations.
//' \item str_return: Indicates whether the result is returned as a string or not.
//' }
//' @field Quantization: From the vector of approximations it looks for its corresponding letter of the alphabet. \itemize{
//' \item one_approx: Vector with the approximations of the time series.
//' }
class SFA {
public:
  SFA(string arg_histogram_type, bool LB = true, bool arg_mft_use_max_or_min = false);
  ~SFA();
  void Initialize(int word_length, int symbols, bool norm_mean);  
  void PrintBins() const;
  vector<std::string> FitTransform(TimeSeries &samples, int word_length, int symbols, bool norm_mean);
  vector<vector<float> > FitTransformDouble(TimeSeries &samples, int word_length, int symbols, bool norm_mean);
  vector<vector<float> > FillOrderline(TimeSeries &samples, int word_length);
  void DivideEquiWidthHistogram(); 
  vector<std::string> Transform(vector<float> &series, vector<vector<float> > approximate); 
  std::string Transform2(vector<float> &series, vector<vector<float> > one_approx, bool str_return = true); 
  vector<int> Quantization(vector<float> &one_approx); 
  std::string SFA2Word(const std::vector<int> &word);  
private:
  bool initialized;
  string histogram_type;
  bool lower_bounding;
  bool mft_use_max_or_min;
  int word_length;
  int max_word_length;
  int symbols;
  bool norm_mean;
  int alphabet_size;
public:
  MFT *transformation;
  vector<vector<float>> *bins;
  vector<vector<std::tuple<float, float> > >  order_line;
};


std::string SFA::SFA2Word(const std::vector<int> &word) {
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



SFA::SFA(string arg_histogram_type, bool LB, bool arg_mft_use_max_or_min) {
  this->initialized = false;
  this->histogram_type = arg_histogram_type;
  this->lower_bounding = LB;
  this->mft_use_max_or_min = arg_mft_use_max_or_min;
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


void SFA::Initialize(int word_length, int symbols, bool arg_norm_mean) {
  this->initialized = true;
  this->word_length = word_length;
  this->max_word_length = word_length;
  this->symbols = symbols;
  this->norm_mean = arg_norm_mean;
  this->alphabet_size = symbols;
  this->transformation = 0;
  
  // Create the buckets for the coefficients
  bins = new vector<vector<float>>(this->alphabet_size);
  vector<float> v(word_length, -INFINITY);
  bins->at(0) = v;

  for (int i = 1; i < this->alphabet_size; ++i) {
    vector<float> v(word_length, INFINITY);
    bins->at(i) = v;
  }
}


void SFA::PrintBins() const {
  // Show buckets
  for (int j = 0; j < this->alphabet_size; ++j) {
    int end_for = static_cast<int>(this->bins[0][j].size());
    for (int k = 0; k < end_for; ++k) {
      cout << this->bins->at(j)[k] << " ";
    }
    cout << endl;
  }
}


vector<std::string> SFA::FitTransform(TimeSeries &samples, int word_length, int symbols, bool norm_mean) {
  // Create the time series approximation
  vector<vector<float> > approximate = this->FitTransformDouble(samples, word_length, symbols, norm_mean);

  // Create the word SFA
  vector<std::string> words = this->Transform(*samples.data, approximate);

  return (words);
}


vector<vector<float> > SFA::FitTransformDouble(TimeSeries &samples, int word_length, int symbols, bool norm_mean) { 
  // Check if the class has been initialized and that the object for the transformation has been created
  if (this->initialized == false) {
    this->Initialize(word_length, symbols, norm_mean);
    if (this->transformation == 0) {
      this->transformation = new MFT(samples.size(), norm_mean, this->lower_bounding);
    }
  }

  // The buckets (bins) are created and the corresponding histogram is called for their division
  vector<vector<float> > transformedSamples = this->FillOrderline(samples, word_length);

  if (this->histogram_type == "EQUI_FREQUENCY") {
    this->DivideEquiWidthHistogram();
  }

  this->order_line.clear();

  return transformedSamples;
}


// Time Series & samples: This parameter should be a vector of TimeSeries if you want to develop the classification algorithms
vector<vector<float> > SFA::FillOrderline(TimeSeries &samples, int word_length) { 
  // Create the tuples containing the fourier coefficient and the string label.
  // The series label is used when developing the classification algorithms
  for (int i = 0; i < word_length; ++i) {
    std::tuple <float, float> t;
    t = std::make_tuple(0.0, 0.0);
    vector<std::tuple<float, float> > vect_tuple;
    vect_tuple.push_back(t);
    this->order_line.push_back(vect_tuple);
  }
  
  // Stores the transformed time series
  vector<vector<float> > transformedSamples;

  // Apply the Fourier transformation to the series
  vector<float> transformedSamples_small = this->transformation->transform(*samples.data, word_length);

  // The transformed series is converted into tuples (pairs)
  transformedSamples.push_back(transformedSamples_small);

  int end_for = static_cast<int>(transformedSamples_small.size());
  for (int j = 0; j < end_for; ++j) {
    float value = std::round(transformedSamples_small[j] * 100.0) / 100.0 ;
    std::tuple <float, float> obj;
    obj = std::make_tuple(value, std::stof(samples.label));
    this->order_line[j][0] = obj;
  }

  // The values of the limits of the buckets are updated based on the transformation
  end_for = static_cast<int>(this->order_line.size());
  for (int i = 0; i < end_for; ++i) {
    vector<std::tuple<float, float> > del_list = this->order_line[i];
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
    this->order_line[i] = new_list;
  }

  return (transformedSamples);
}


void SFA::DivideEquiWidthHistogram() {
  // All pairs of buckets are traversed
  int end_for = static_cast<int>(this->order_line.size());
  for (int i = 0; i < end_for; ++i) {
    vector<std::tuple<float, float> > element = this->order_line[i];

    // Make all gaps between buckets equal to a fixed size
    if (element.size() > 0) {
      float first = std::get<0>(element.front());
      float last = std::get<0>(element.back());
      float intervalWidth = (last - first) / this->alphabet_size;
      int end_for2 = this->alphabet_size - 1;
      for (int j = 0; j < end_for2; ++j) {
        this->bins->at(j+1)[i] = intervalWidth * (j + 1) + first;
      }
    }
  }
}


vector<std::string> SFA::Transform(vector<float> &series, vector<vector<float> > approximate) {
  vector<std::string> words;
  
  words.push_back(this->Transform2(series, approximate)); 
  
  return (words);
}


std::string SFA::Transform2(vector<float> &series, vector<vector<float> > one_approx, bool str_return) {
  vector<float> one_approx_2;
  std::string word;
  
  // Check if the bucket approximation exists or not.
  if (one_approx.size() == 0) {
    one_approx_2 = this->transformation->transform(series, this->max_word_length);
    
    if (str_return) {
      word = this->SFA2Word(this->Quantization(one_approx_2));
    } else {
      //return (this->Quantization(one_approx));  
    }
  } else {
    if (str_return) {
      word = this->SFA2Word(this->Quantization(one_approx[0]));
    } else {
      //return (this->Quantization(one_approx));  
    }
  }
  
  return (word);
}  


vector<int> SFA::Quantization(vector<float> &one_approx) {
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
bool ValidateParameters(const std::string histogram_type, int word_length, int symbols, int norm_mean, int max_window_length, int sliding_window, int serie_size) {
  string error_message = "Errors in parameters: ";
  bool valid_parameters = true;
  
  // histogram_type
  bool found = false;
  std::vector<std::string> histogram_types = {"INFORMATION_GAIN", "EQUI_FREQUENCY"};
  found = std::find(histogram_types.begin(), histogram_types.end(), histogram_type) != histogram_types.end();
  if (!found) {
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


//' @name fn_sfa2 function
//' @title Constructs a new Double object
//' @param serie time serie to convert to word.
//' @param sw time se convert to word.
//' @param wl tim serie to convert to word.
//' @param as time see to convert to word.
//' @param norm_mean ime serie to convert to word.
//' @param lower_bounding time se convert to word.
//' @param mft_use_max_or_min tito convert to word.
//' @param norm_check time serie to c to word.
//' @param apply_z_norm time serie to conveword.
//' @param normalize time serie to convert to d.
//' @return SFA words
// [[Rcpp::export("fn_sfa")]]
CharacterVector fn_sfa2(NumericVector data, int sliding_window = 120, int word_length = 6, int alphabet_size = 5, bool norm_mean = true, bool apply_z_norm = true) {
  bool lower_bounding = true;
  bool mft_use_max_or_min = false;
  bool norm_check = true;

  // int sliding_window_aux = Rcpp::as<int>(sliding_window); 
  vector<float> serie_aux = Rcpp::as< std::vector<float> >(data);

  // ------------------------------ Read and validate the parameters ------------- -----------------
  std::string histogram_type = "EQUI_FREQUENCY";
  int symbols = alphabet_size; 
  int max_window_length = word_length;
  string label = "1";

  bool validated_parameters = ValidateParameters(histogram_type, word_length, symbols, norm_mean, max_window_length, sliding_window, (int)serie_aux.size());
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

    // serie_train_obj = new TimeSeries(serie_sample, label, norm_check, apply_z_norm);
    TimeSeries serie_train_obj(serie_sample, label, norm_check, apply_z_norm);
    if (apply_z_norm) {
      serie_train_obj.Norm(norm_mean);
    }

    // Perform the transformation for the sliding window
    SFA sfa_object(histogram_type, lower_bounding, mft_use_max_or_min);

    words_sfa.at(i) = sfa_object.FitTransform(serie_train_obj, word_length, symbols, norm_mean)[0];
  }

  return (Rcpp::wrap(words_sfa));
}
