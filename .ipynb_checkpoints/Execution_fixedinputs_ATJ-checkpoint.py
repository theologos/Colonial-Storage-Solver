{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "2fa77fde",
   "metadata": {},
   "source": [
    "# Define the model fixed inputs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "28e7432f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2fa9016d",
   "metadata": {},
   "source": [
    "## Time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "202c03bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "T         = 120\n",
    "Time      = list(range(T))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "df814e37",
   "metadata": {},
   "source": [
    "## Bounds"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "bbcf4c22",
   "metadata": {},
   "outputs": [],
   "source": [
    "Bounds   = {\n",
    "            \"01\": {\"l\": 25,   \"u\": 28},\n",
    "            \"02\": {\"l\": 6,    \"u\": 10},\n",
    "            \"13\": {\"l\": 2,    \"u\": 2},\n",
    "            \"14\": {\"l\": 2,    \"u\": 2.4}, \n",
    "            \"15\": {\"l\": 4,    \"u\": 5},\n",
    "            \"16\": {\"l\": 2,    \"u\": 2.15},\n",
    "            \"17\": {\"l\": 4,    \"u\": 5},\n",
    "            \"18\": {\"l\": 7,    \"u\": 8.7},\n",
    "            \"19\": {\"l\": 4,    \"u\": 5.1},\n",
    "            \"20\": {\"l\": 1,    \"u\": 1.75}\n",
    "           }"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4ec593d1",
   "metadata": {},
   "source": [
    "## Capacities & Tanks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "5d127851",
   "metadata": {},
   "outputs": [],
   "source": [
    "filename = 'data_new_1/capacities_ATJ.csv'\n",
    "Capacity = {}\n",
    "df1 = pd.read_csv (filename, encoding = \"ISO-8859-1\")\n",
    "for index, row in df1.iterrows():\n",
    "    Capacity[int(row['Tank'])] = round(int(1000 * row['Working'])/1000)\n",
    "Capacity[336]  = 115000/1000\n",
    "Capacity[361]  = 120000/1000\n",
    "Capacity[371]  = 120000/1000\n",
    "Capacity[373]  = 200000/1000\n",
    "\n",
    "Tanks = list(Capacity.keys())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9a47311b",
   "metadata": {},
   "source": [
    "## Topologies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "0c8668bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "topo_i = {\n",
    "          311: {'01': {'A': 0}},\n",
    "          312: {'01': {'D': 0}},\n",
    "          313: {'01': {'A': 0}},\n",
    "          314: {'01': {'A': 0}},\n",
    "          316: {'01': {'D': 0}},\n",
    "          317: {'01': {'A': 0}},\n",
    "          330: {'01': {'A': 0}},\n",
    "          331: {'01': {'A': 0}},\n",
    "          332: {'01': {'A': 0}},\n",
    "          333: {'01': {'A': 0}},\n",
    "          334: {'01': {'A': 0}},\n",
    "          336: {'01': {'A': 0}},\n",
    "          337: {'01': {'D': 0}},\n",
    "          338: {'01': {'A': 0}},\n",
    "          339: {'01': {'A': 0}},\n",
    "          350: {'02': {'54': 0}},\n",
    "          351: {'02': {'54': 0}},\n",
    "          352: {'02': {'54': 0}},\n",
    "          353: {'02': {'54': 0}},\n",
    "          354: {'02': {'54': 0}},\n",
    "          361: {'02': {'62': 0}},\n",
    "          363: {'02': {'62': 0}},\n",
    "          371: {'02': {'62': 0}},\n",
    "          374: {'02': {'62': 0}},\n",
    "          375: {'02': {'62': 0}},\n",
    "          376: {'02': {'62': 0}}\n",
    "}\n",
    "\n",
    "# topo_o = {\n",
    "#           311: {'15': {'A': 0}},\n",
    "#           312: {'15': {'D': 0}, '14': {'D': 0}, '17': {'D': 0}},\n",
    "#           313: {'17': {'A': 0}},\n",
    "#           314: {'15': {'A': 0}},\n",
    "#           316: {'18': {'D': 0}, '19': {'D': 0}, '17': {'D': 0}},\n",
    "#           317: {'14': {'A': 0}},\n",
    "#           330: {'18': {'A': 0}},\n",
    "#           331: {'18': {'A': 0}},\n",
    "#           332: {'17': {'A': 0}, '18': {'A': 0}},\n",
    "#           333: {'19': {'A': 0}},\n",
    "#           334: {'15': {'A': 0}},      \n",
    "#           336: {'19': {'A': 0}},\n",
    "#           337: {'18': {'D': 0}},    \n",
    "#           #338: {'19': {'A': 0}}, #####\n",
    "#           339: {'18': {'A': 0}},\n",
    "#           350: {'20': {'54': 0}},\n",
    "#           351: {'16': {'54': 0}},\n",
    "#           352: {'16': {'54': 0}},\n",
    "#           353: {'16': {'54': 0}},\n",
    "#           354: {'18': {'54': 0}, '16': {'54': 0}, '20': {'54': 0}},\n",
    "#           361: {'13': {'62': 0}},\n",
    "#           #371: {'13': {'62': 0}, '18': {'62': 0}}, #####\n",
    "#           374: {'14': {'62': 0}, '17': {'62': 0}, '20': {'62': 0}},\n",
    "#           375: {'18': {'62': 0}},\n",
    "#           376: {'18': {'62': 0}}\n",
    "# }\n",
    "\n",
    "topo_o = {\n",
    "          311: {'13': {'A': 0}, '14': {'A': 0}, '15': {'A': 0}, '17': {'A': 0}, '18': {'A': 0}, '19': {'A': 0}},\n",
    "          312: {'13': {'D': 0}, '14': {'D': 0}, '15': {'D': 0}, '17': {'D': 0}, '18': {'D': 0}, '19': {'D': 0}},\n",
    "          313: {'13': {'A': 0}, '14': {'A': 0}, '15': {'A': 0}, '17': {'A': 0}, '18': {'A': 0}, '19': {'A': 0}},\n",
    "          314: {'13': {'A': 0}, '14': {'A': 0}, '15': {'A': 0}, '17': {'A': 0}, '18': {'A': 0}, '19': {'A': 0}},\n",
    "          316: {'13': {'D': 0}, '14': {'D': 0}, '15': {'D': 0}, '17': {'D': 0}, '18': {'D': 0}, '19': {'D': 0}},\n",
    "          317: {'13': {'A': 0}, '14': {'A': 0}, '15': {'A': 0}, '17': {'A': 0}, '18': {'A': 0}, '19': {'A': 0}},\n",
    "          330: {'13': {'A': 0}, '14': {'A': 0}, '15': {'A': 0}, '17': {'A': 0}, '18': {'A': 0}, '19': {'A': 0}},\n",
    "          331: {'13': {'A': 0}, '14': {'A': 0}, '15': {'A': 0}, '17': {'A': 0}, '18': {'A': 0}, '19': {'A': 0}},\n",
    "          332: {'13': {'A': 0}, '14': {'A': 0}, '15': {'A': 0}, '17': {'A': 0}, '18': {'A': 0}, '19': {'A': 0}},\n",
    "          333: {'13': {'A': 0}, '14': {'A': 0}, '15': {'A': 0}, '17': {'A': 0}, '18': {'A': 0}, '19': {'A': 0}},\n",
    "          334: {'13': {'A': 0}, '14': {'A': 0}, '15': {'A': 0}, '17': {'A': 0}, '18': {'A': 0}, '19': {'A': 0}},     \n",
    "          336: {'13': {'A': 0}, '14': {'A': 0}, '15': {'A': 0}, '17': {'A': 0}, '18': {'A': 0}, '19': {'A': 0}},\n",
    "          337: {'13': {'D': 0}, '14': {'D': 0}, '15': {'D': 0}, '17': {'D': 0}, '18': {'D': 0}, '19': {'D': 0}},   \n",
    "          338: {'13': {'A': 0}, '14': {'A': 0}, '15': {'A': 0}, '17': {'A': 0}, '18': {'A': 0}, '19': {'A': 0}},\n",
    "          339: {'13': {'A': 0}, '14': {'A': 0}, '15': {'A': 0}, '17': {'A': 0}, '18': {'A': 0}, '19': {'A': 0}},\n",
    "          350: {'13': {'54': 0}, '14': {'54': 0}, '16': {'54': 0}, '17': {'54': 0}, '18': {'54': 0}, '19': {'54': 0}, '20': {'54': 0}},\n",
    "          351: {'13': {'54': 0}, '14': {'54': 0}, '16': {'54': 0}, '17': {'54': 0}, '18': {'54': 0}, '19': {'54': 0}, '20': {'54': 0}},\n",
    "          352: {'13': {'54': 0}, '14': {'54': 0}, '16': {'54': 0}, '17': {'54': 0}, '18': {'54': 0}, '19': {'54': 0}, '20': {'54': 0}},\n",
    "          353: {'13': {'54': 0}, '16': {'54': 0}, '17': {'54': 0}, '18': {'54': 0}, '19': {'54': 0}},\n",
    "          354: {'16': {'54': 0}, '18': {'54': 0}, '20': {'54': 0}},\n",
    "          361: {'13': {'62': 0}, '14': {'62': 0}, '17': {'62': 0}, '18': {'62': 0}, '19': {'62': 0}, '20': {'62': 0}},\n",
    "          363: {'13': {'62': 0}, '14': {'62': 0}, '17': {'62': 0}, '18': {'62': 0}, '19': {'62': 0}, '20': {'62': 0}},\n",
    "          371: {'13': {'62': 0}, '14': {'62': 0}, '17': {'62': 0}, '18': {'62': 0}, '19': {'62': 0}, '20': {'62': 0}},\n",
    "          374: {'13': {'62': 0}, '14': {'62': 0}, '17': {'62': 0}, '18': {'62': 0}, '19': {'62': 0}, '20': {'62': 0}},\n",
    "          375: {'13': {'62': 0}, '14': {'62': 0}, '17': {'62': 0}, '18': {'62': 0}, '19': {'62': 0}, '20': {'62': 0}},\n",
    "          376: {'13': {'62': 0}, '14': {'62': 0}, '17': {'62': 0}, '18': {'62': 0}, '19': {'62': 0}, '20': {'62': 0}}\n",
    "}"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}