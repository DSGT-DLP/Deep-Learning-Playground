import * as React from 'react'
import { FaCloudUploadAlt } from 'react-icons/fa'
import { useDispatch, useSelector } from 'react-redux'
import { RootState } from '../../redux/store'
import { handleFileUpload } from '../../redux/train' /*}

/*const CSVInputFile2 = () => {
  const fileName = useSelector<RootState>((state) => state.train.fileName);
  const dispatch = useDispatch();
  return (
    <div id="CSVInputFile2">
      {/*       <label
        htmlFor="csv-upload"
        className="custom-file-upload d-flex align-items-center"
      >
        <>
          <FaCloudUploadAlt className="me-2" />
          {fileName ?? "Upload CSV"}
        </>
      </label> */
/*<input
        type="file"
        id="csv-upload"
        accept=".csv,.xlsx,.xls"
        onChange={(e) => dispatch(handleFileUpload(e.target.files[0]))}
        style={{ width: "100%" }}
      />
    </div>
  );
};*/

/*export default CSVInputFile2;*/
