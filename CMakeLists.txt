CMAKE_MINIMUM_REQUIRED( VERSION 3.0 )

SET( CMAKE_EXPORT_COMPILE_COMMANDS 1 )

PROJECT( NeuralNetwork C )

INCLUDE_DIRECTORIES( include )
ADD_SUBDIRECTORY( src )

ENABLE_TESTING()
ADD_SUBDIRECTORY( tests )


FIND_PACKAGE( Doxygen )
IF( DOXYGEN_FOUND )
        CONFIGURE_FILE(
                ${PROJECT_SOURCE_DIR}/Doxyfile.in
                ${PROJECT_BINARY_DIR}/Doxyfile @ONLY)
        ADD_CUSTOM_TARGET(
                docs
                ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/Doxyfile
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                COMMENT "Generating Documentation with Doxygen" VERBATIM)
ENDIF( DOXYGEN_FOUND )
