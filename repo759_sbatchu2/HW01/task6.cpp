#include <iostream>
using namespace std;

//argv is a string array(char**). Each string is pointed with char*. The string has to be converted to number.
int string_to_integer(char* x)
{
        int i=0;
        int number = 0;
        while(x[i]!='\0'){
          number = number*10 + (x[i] - 48);
          i++;
        }
  return number;
}

int main(int argc, char** argv)
{
  char* N_string = argv[1]; // First argument, 0 is executable
  int N = atoi(N_string);
  //int N = string_to_integer(N_string); //(uncomment if local string to integer function need to be use)
  //cout<<N<<endl;
  
  //Print 0 to N
  for(int i=0; i<=N; i++)
	  cout <<i<<" ";
  cout<<endl;
  for(int i=N; i>=0; i--)
	  cout <<i<<" ";

  return 0;
}
