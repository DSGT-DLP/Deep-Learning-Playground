import "chartjs-adapter-date-fns";
import { ChartData } from "chart.js";
import { enUS } from "date-fns/locale";
import React, { useEffect, useState } from "react";
import { Bar } from "react-chartjs-2";
import { TrainSpaceData } from "../types/trainTypes";
import { add, format, isFuture } from "date-fns";
import {
  Chart as ChartJS,
  Tooltip,
  Legend,
  LinearScale,
  BarElement,
  Title,
  TimeSeriesScale,
} from "chart.js";
ChartJS.register(
  Tooltip,
  Legend,
  LinearScale,
  BarElement,
  Title,
  TimeSeriesScale
);
const TrainBarChart = ({
  trainSpaceDataArr,
}: {
  trainSpaceDataArr?: TrainSpaceData[];
}) => {
  const [execFrequencyBarData, setExecFrequencyBarData] = useState<ChartData<
    "bar",
    { x: Date; y: number }[]
  > | null>(null);
  useEffect(() => {
    if (trainSpaceDataArr) {
      const sameDay = (d1: Date, d2: Date) => {
        return (
          d1.getFullYear() === d2.getFullYear() &&
          d1.getMonth() === d2.getMonth() &&
          d1.getDate() === d2.getDate()
        );
      };
      const setToNearestDay = (d: Date) => {
        d.setHours(0, 0, 0, 0);
        return d;
      };
      const execFrequencyData: { x: Date; y: number }[] = [];
      trainSpaceDataArr.forEach((row) => {
        if (isFuture(add(new Date(row.timestamp), { days: 30 }))) {
          execFrequencyData.length !== 0 &&
          sameDay(
            new Date(row.timestamp),
            execFrequencyData[execFrequencyData.length - 1].x
          )
            ? (execFrequencyData[execFrequencyData.length - 1].y += 1)
            : execFrequencyData.push({
                x: setToNearestDay(new Date(row.timestamp)),
                y: 1,
              });
        }
      });
      setExecFrequencyBarData({
        datasets: [
          {
            label: "# of Executions",
            backgroundColor: "rgba(75, 192, 192, 0.7)",
            borderColor: "rgb(75, 192, 192)",
            borderWidth: 1,
            barThickness: 15,
            data: execFrequencyData,
          },
        ],
      });
    }
  }, [trainSpaceDataArr]);
  return (
    <>
      {execFrequencyBarData ? (
        <Bar
          data={execFrequencyBarData}
          options={{
            maintainAspectRatio: false,
            scales: {
              x: {
                adapters: {
                  date: {
                    locale: enUS,
                  },
                },
                ticks: {
                  maxRotation: 80,
                  minRotation: 80,
                },
                type: "timeseries",
                time: {
                  unit: "day",
                  minUnit: "day",
                  displayFormats: {
                    day: "MMM dd",
                  },
                },
              },
              y: {
                beginAtZero: true,
              },
            },
            responsive: true,
            plugins: {
              tooltip: {
                callbacks: {
                  title: (context) => {
                    return format(
                      execFrequencyBarData.datasets[0].data[
                        context[0].dataIndex
                      ].x,
                      "MMM d"
                    );
                  },
                },
              },
              legend: {
                display: false,
              },
              title: {
                display: true,
                text: "Training Frequency",
              },
            },
          }}
        />
      ) : null}
    </>
  );
};

export default TrainBarChart;
