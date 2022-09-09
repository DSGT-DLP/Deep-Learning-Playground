import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from "@mui/material";
import React from "react";
import { useNavigate } from "react-router-dom";

const rows = [
  // {
  //   id: "as98dfumasdp",
  //   type: "Tabular",
  //   input: "my_tabular_input.csv",
  //   status: "35%",
  //   date: "Sep 4, 2022",
  // },
  // {
  //   id: "p9umaspdf",
  //   type: "Image Training",
  //   input: "my_images.zip",
  //   status: "Done",
  //   date: "Sep 1, 2022",
  // },
];

const BlankGrid = () => {
  const navigate = useNavigate();

  return (
    <div id="blank-grid-wrapper">
      <div id="blank-grid">
        <p>
          You haven't trained any models yet. Create your first model below!
        </p>
        <button id="blank-grid-button" onClick={() => navigate("/")}>
          Train Model
        </button>
      </div>
    </div>
  );
};

const FilledGrid = () => {
  return (
    <TableContainer>
      <Table sx={{ minWidth: 400 }}>
        <TableHead>
          <TableRow>
            <TableCell>Type</TableCell>
            <TableCell align="right">Input</TableCell>
            <TableCell align="right">Status</TableCell>
            <TableCell align="right">Date</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row) => (
            <TableRow
              key={row.id}
              sx={{ "&:last-child td, &:last-child th": { border: 0 } }}
            >
              <TableCell component="th" scope="row">
                {row.type}
              </TableCell>
              <TableCell align="right">{row.input}</TableCell>
              <TableCell align="right">{row.status}</TableCell>
              <TableCell align="right">{row.date}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

const Dashboard = () => {
  return (
    <div id="dashboard">
      <FilledGrid />
      {rows.length === 0 && <BlankGrid />}
    </div>
  );
};

export default Dashboard;
