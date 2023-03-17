import React from 'react'
import PropTypes from 'prop-types'
import { COLORS } from '../../constants'

const TitleText = (props) => {
  const { text } = props
  return <h2 style={styles.titleText}>{text}</h2>
}

export default TitleText

const styles = {
  titleText: { color: COLORS.layer },
}

TitleText.propTypes = {
  text: PropTypes.string,
}
