#include "Token.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  CTORs/DTORs
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Token::Token()
{
	m_contents = {};
	m_type = Token::TokenType::TOKEN_NULL;
}

Token::Token(Token::TokenType type, std::vector<unsigned char> currentchars)
{
	m_contents = currentchars;
	m_type = type;
}


Token::~Token()
{

}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Function toString()
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string Token::toString(Token::TokenType tt)
{

	switch (tt) {

	case Token::TokenType::TOKEN_WHITESPACE: return "TOKEN_WHITESPACE"; break;
	case Token::TokenType::TOKEN_ID: return "TOKEN_ID"; break;
	case Token::TokenType::TOKEN_EQUALS: return "TOKEN_EQUALS"; break;
	case Token::TokenType::TOKEN_STRING: return "TOKEN_STRING"; break;
	case Token::TokenType::TOKEN_SEMI: return "TOKEN_SEMI"; break;
	case Token::TokenType::TOKEN_LPAREN: return "TOKEN_LPAREN"; break;
	case Token::TokenType::TOKEN_RPAREN: return "TOKEN_RPAREN"; break;
	case Token::TokenType::TOKEN_LBRACE: return "TOKEN_LBRACE"; break;
	case Token::TokenType::TOKEN_RBRACE: return "TOKEN_RBRACE"; break;
	case Token::TokenType::TOKEN_LBRACKET: return "TOKEN_LBRACKET"; break;
	case Token::TokenType::TOKEN_RBRACKET: return "TOKEN_RBRACKET"; break;
	case Token::TokenType::TOKEN_COLON: return "TOKEN_COLON"; break;
	case Token::TokenType::TOKEN_COMMA: return "TOKEN_COMMA"; break;
	case Token::TokenType::TOKEN_BACKWARD_SLASH: return "TOKEN_BACKWARD_SLASH"; break;
	case Token::TokenType::TOKEN_FORWARD_SLASH: return "TOKEN_FORWARD_SLASH"; break;
	case Token::TokenType::TOKEN_POUND: return "TOKEN_POUND"; break;
	case Token::TokenType::TOKEN_LT: return "TOKEN_LT"; break;
	case Token::TokenType::TOKEN_GT: return "TOKEN_GT"; break;
	case Token::TokenType::TOKEN_ASTERISK: return "TOKEN_ASTERISK"; break;
	case Token::TokenType::TOKEN_MINUS: return "TOKEN_MINUS"; break;
	case Token::TokenType::TOKEN_PLUS: return "TOKEN_PLUS"; break;
	case Token::TokenType::TOKEN_EXCLAMATION_MARK: return "TOKEN_EXCLAMATION_MARK"; break;
	case Token::TokenType::TOKEN_PERIOD: return "TOKEN_PERIOD"; break;
	case Token::TokenType::TOKEN_QUESTION_MARK: return "TOKEN_QUESTION_MARK"; break;
	case Token::TokenType::TOKEN_CARET: return "TOKEN_CARET"; break;
	case Token::TokenType::TOKEN_AT: return "TOKEN_AT"; break;
	case Token::TokenType::TOKEN_PERCENT: return "TOKEN_PERCENT"; break;
	case Token::TokenType::TOKEN_AMPERSAND: return "TOKEN_AMPERSAND"; break;
	case Token::TokenType::TOKEN_PIPE: return "TOKEN_PIPE"; break;
	case Token::TokenType::TOKEN_CHAR: return "TOKEN_CHAR"; break;
	case Token::TokenType::TOKEN_INT: return "TOKEN_INT"; break;
	case Token::TokenType::TOKEN_DOLLAR: return "TOKEN_DOLLAR"; break;
	case Token::TokenType::TOKEN_TILDE: return "TOKENTOKEN_TILDE_INT"; break;
	case Token::TokenType::TOKEN_BACKTICK: return "TOKEN_BACKTICK"; break;
	case Token::TokenType::TOKEN_SINGLE_QUOTE: return "TOKEN_SINGLE_QUOTE"; break;
	case Token::TokenType::TOKEN_DOUBLE_QUOTE: return "TOKEN_DOUBLE_QUOTE"; break;
	case Token::TokenType::TOKEN_UTF8: return "TOKEN_UTF8"; break;		
	case Token::TokenType::TOKEN_EOF: return "TOKEN_EOF"; break;
	default: return "undefined"; break;

	}
}