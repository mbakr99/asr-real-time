#include "decoders/lexicon.hpp"

int main(int argc, char** argv){
    
    // set direcotry to lexicon  
    auto project_root_ptr = std::getenv("PROJECT_ROOT");
    fs::path project_root = fs::path(project_root_ptr);
    fs::path lexicon_dir  = project_root / "data" / "lexicon"; 
    fs::path lexicon_file_path  = lexicon_dir / "lexicon.txt";
    fs::path lexicon_fst_target = lexicon_dir / "lexicon_fst.fst";

    // build and save lexicon 
    LexiconFst lexicon_builder(lexicon_file_path);
    lexicon_builder.construct_fst_from_lex_file();
    lexicon_builder.write_fst(lexicon_fst_target);


}