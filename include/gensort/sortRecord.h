#ifndef __SORT_RECORD_H_
#define __SORT_RECORD_H_

class sortRecord {
private:
  /*
  long      eka;
  short     dva;
  */
  char      key[10];
  char      value[90];
public:
  // TODO : optimize using SIMD
  bool  operator == ( sortRecord const  &other) const {
    return (memcmp(this->key, other.key, 10) == 0);
    // return ( (this->eka == other.eka) && (this->dva == other.dva) );
    // return ( (*(unsigned long*)(this->key) == *(unsigned long*)(other.key)) &&  (*(unsigned short*)(this->key+8) == *(unsigned short*)(other.key+8)) );
  }
  bool  operator < ( sortRecord const  &other) const {
    return (memcmp(this->key, other.key, 10) < 0);
    
    /*
    if (*(long*)(this->key) == *(long*)(other.key)) 
      return (*(unsigned short*)(this->key+8) < *(unsigned short*)(other.key+8)); 
    else
      return (*(unsigned long*)(this->key) < *(unsigned long*)(other.key)); 
    if (this->eka == other.eka)
      return this->dva < other.dva;
    else
      return this->eka < other.eka;
      */
  }
  bool  operator <= ( sortRecord const  &other) const {
    return (memcmp(this->key, other.key, 10) <= 0);
  }
  bool  operator > ( sortRecord const  &other) const {
    return (memcmp(this->key, other.key, 10) > 0);
  }
  bool  operator >= ( sortRecord const  &other) const {
    return (memcmp(this->key, other.key, 10) >= 0);
  }
}; 


namespace par {

  //Forward Declaration
  template <typename T>
    class Mpi_datatype;

  template <>
    class Mpi_datatype< sortRecord > {

      public:

      /**
       @return The MPI_Datatype corresponding to the datatype "sortRecord".
     */
      static MPI_Datatype value()
      {
        static bool         first = true;
        static MPI_Datatype datatype;

        if (first)
        {
          first = false;
          MPI_Type_contiguous(sizeof(sortRecord), MPI_BYTE, &datatype);
          MPI_Type_commit(&datatype);
        }

        return datatype;
      }

    };

}//end namespace par


#endif
